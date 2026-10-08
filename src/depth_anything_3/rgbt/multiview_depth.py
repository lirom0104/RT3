"""Source-only depth search, boundary-preserving consensus and Gaussian correction.

The search does not reject a correspondence because it disagrees with the old
DA3 depth. All candidates are checked against independent source image evidence.
Target cameras, target pixels and external reconstructed depths are never inputs.
"""

from dataclasses import replace
import math
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import gaussian_depth


def project(points, E, K, height, width):
    camera = points @ E[:3, :3].T + E[:3, 3]
    uvz = camera @ K.T
    uv = uvz[..., :2] / uvz[..., 2:].clamp_min(1e-6)
    grid = 2 * uv / uv.new_tensor([width, height]) - 1
    return grid, camera[..., 2]


def sample(image, grid):
    return F.grid_sample(image[None], grid[None], align_corners=False)[0]


class MultiViewDepthCorrector(nn.Module):
    """A fixed photometric cost volume with a trainable confidence/residual head."""

    def __init__(
        self,
        hypotheses=129,
        min_ratio=0.4,
        max_ratio=2.5,
        chunk=16,
        channels=32,
        learned=False,
        gate_threshold=0.004,
        apply_before=False,
        visibility_strength=0.0,
        line_support=True,
        min_improvement=0.0,
        min_margin=0.0,
        min_support=0.15,
        hard_gate=False,
        source_transfer=False,
        conflict_strength=0.0,
        wide_consistency_min_sources=1,
    ):
        super().__init__()
        if hypotheses < 5 or not 0 < min_ratio < 1 < max_ratio:
            raise ValueError(
                "Depth search must bracket the prior with positive hypotheses"
            )
        self.config = dict(
            hypotheses=hypotheses,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            chunk=chunk,
            channels=channels,
            learned=learned,
            gate_threshold=gate_threshold,
            apply_before=apply_before,
            visibility_strength=visibility_strength,
            line_support=line_support,
            min_improvement=min_improvement,
            min_margin=min_margin,
            min_support=min_support,
            hard_gate=hard_gate,
            source_transfer=source_transfer,
            conflict_strength=conflict_strength,
            wide_consistency_min_sources=wide_consistency_min_sources,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(13, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 3, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.last_audit = None

    @torch.no_grad()
    def prepare(self, rgb, initial, source_valid=None):
        if rgb.ndim != 5 or rgb.shape[0] != 1:
            raise ValueError("Expected one scene of source RGB images")
        b, v, _, h, w = rgb.shape
        if initial.depth.shape != (b, v, h, w):
            raise ValueError("Source images and Gaussian grids must agree")
        valid = (
            torch.ones_like(rgb[:, :, 0])
            if source_valid is None
            else source_valid[:, :, 0].float()
        )
        if v < 2:
            z = gaussian_depth(initial).float()
            return dict(
                depth=z,
                prior=z,
                confidence=torch.zeros_like(z),
                margin=torch.zeros_like(z),
                cost=torch.zeros_like(z),
                improvement=torch.zeros_like(z),
                support=torch.zeros_like(z),
                disagreement=torch.zeros_like(z),
                valid=valid,
            )
        with torch.autocast(rgb.device.type, enabled=False):
            rgb = rgb[0].float()
            valid = valid[0]
            E = initial.extrinsics[0].float()
            K = initial.intrinsics[0].float()
            prior = gaussian_depth(initial)[0].float().clamp_min(1e-5)
            means = initial.gaussians.means[0].reshape(v, h, w, 3).float()
            centers = torch.linalg.inv(E)[:, :3, 3]
            # Preserve actual subpixel Gaussian anchors when replacing camera Z.
            rays = (means - centers[:, None, None]) / prior[..., None]
            contrast = rgb - F.avg_pool2d(rgb, 9, 1, 4)
            features = torch.cat([rgb, 2 * contrast], 1)
            ratios = torch.exp(
                torch.linspace(
                    math.log(self.config["min_ratio"]),
                    math.log(self.config["max_ratio"]),
                    self.config["hypotheses"],
                    device=rgb.device,
                )
            )
            ratios = torch.cat([ratios, ratios.new_ones(1)]).sort().values
            depths = []
            margins = []
            costs = []
            improvements = []
            counts = []
            for i in range(v):
                volumes = []
                coverage = []
                for rs in ratios.split(self.config["chunk"]):
                    xyz = (
                        centers[i]
                        + rays[i][None]
                        * (prior[i][None] * rs[:, None, None])[..., None]
                    )
                    pair = []
                    masks = []
                    for j in range(v):
                        if i == j:
                            continue
                        grid, z = project(xyz, E[j], K[j], h, w)
                        warped = F.grid_sample(
                            features[j : j + 1].expand(len(rs), -1, -1, -1),
                            grid,
                            align_corners=False,
                        )
                        error = (warped - features[i][None]).abs().mean(1)
                        error = (
                            0.6 * error
                            + 0.4 * F.avg_pool2d(error[:, None], 3, 1, 1)[:, 0]
                        )
                        ok = (grid.abs() < 0.985).all(-1) & (z > 0)
                        ok &= (
                            F.grid_sample(
                                valid[j : j + 1, None].expand(len(rs), -1, -1, -1),
                                grid,
                                align_corners=False,
                            )[:, 0]
                            > 0.99
                        )
                        pair.append(error.masked_fill(~ok, float("inf")))
                        masks.append(ok)
                    pair = torch.stack(pair)
                    mask = torch.stack(masks)
                    # Two agreeing sources suffice; farther cameras may be occluded.
                    k = min(2, v - 1)
                    best = pair.topk(k, dim=0, largest=False).values
                    count = mask.sum(0)
                    denom = count.clamp(1, k)
                    cost = best.masked_fill(~torch.isfinite(best), 0).sum(0) / denom
                    cost = cost.masked_fill(count == 0, 1.0)
                    volumes.append(cost)
                    coverage.append(count)
                cost = torch.cat(volumes)
                support = torch.cat(coverage)
                score = cost + 0.005 * ratios.log().abs()[:, None, None]
                best, index = score.min(0)
                # Parabolic sub-bin refinement; never average distinct depth modes.
                below = (index - 1).clamp(0, len(ratios) - 1)
                above = (index + 1).clamp(0, len(ratios) - 1)
                a = score.gather(0, below[None])[0]
                c = score.gather(0, above[None])[0]
                delta = (0.5 * (a - c) / (a - 2 * best + c).clamp_min(1e-6)).clamp(
                    -0.5, 0.5
                )
                step = (ratios[above].log() - ratios[below].log()) * 0.5
                depth = prior[i] * (ratios[index].log() + delta * step).exp()
                separated = (
                    ratios[:, None, None].log() - ratios[index].log()[None]
                ).abs() > 0.10
                runner = score.masked_fill(~separated, 2.0).amin(0)
                margin = (runner - best).clamp_min(0)
                base = cost[(ratios - 1).abs().argmin()]
                depths.append(depth)
                margins.append(margin)
                costs.append(best)
                improvements.append((base - best).clamp_min(0))
                counts.append(support.gather(0, index[None])[0])
            depth = torch.stack(depths)
            margin = torch.stack(margins)
            cost = torch.stack(costs)
            improvement = torch.stack(improvements)
            counts = torch.stack(counts)
            points = centers[:, None, None] + rays * depth[..., None]
            agreements = []
            disagreements = []
            for i in range(v):
                checks = []
                bad = []
                for j in range(v):
                    if i == j:
                        continue
                    grid, z = project(points[i], E[j], K[j], h, w)
                    other = sample(depth[j, None], grid)[0]
                    other_margin = sample(margin[j, None], grid)[0]
                    visible = (
                        (grid.abs() < 0.985).all(-1)
                        & (z > 0)
                        & (sample(valid[j, None], grid)[0] > 0.99)
                    )
                    logerr = (z.clamp_min(1e-6) / other.clamp_min(1e-6)).log().abs()
                    checks.append(torch.exp(-logerr.square() / (2 * 0.04**2)) * visible)
                    # Absence/occlusion alone is not contrary evidence.
                    bad.append(
                        (logerr > 0.12)
                        * visible
                        * (other_margin > 0.008)
                        * (z < other * 1.05)
                    )
                agreements.append(torch.stack(checks).amax(0))
                disagreements.append(torch.stack(bad).float().mean(0))
            agreement = torch.stack(agreements)
            disagreement = torch.stack(disagreements)
            strength = (margin / self.config["gate_threshold"]).clamp(0, 1)
            strength *= torch.exp(-cost.square() / (2 * 0.16**2))
            strength *= agreement
            strength *= valid * (counts > 0)
            # Treat a trusted match as a surface hypothesis, not a blend of two surfaces.
            confidence = strength
            # Bilateral averaging of inverse depth within the same estimated surface.
            # High-confidence neighbors repair isolated mismatches without crossing RGB edges.
            inv = depth.reciprocal()
            total = torch.zeros_like(inv)
            mass = torch.zeros_like(inv)
            for dy, dx in [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]:
                neighbor = torch.roll(inv, (dy, dx), (-2, -1))
                cc = torch.roll(confidence, (dy, dx), (-2, -1))
                color = torch.roll(rgb, (dy, dx), (-2, -1))
                weight = cc * torch.exp(-(color - rgb).square().mean(1) / (0.06**2))
                weight *= torch.exp(
                    -((neighbor / inv.clamp_min(1e-6)).log()).square() / (2 * 0.10**2)
                )
                weight *= math.exp(-(dy * dy + dx * dx) / 8)
                if dy > 0:
                    weight[:, :dy] = 0
                elif dy < 0:
                    weight[:, dy:] = 0
                if dx > 0:
                    weight[:, :, :dx] = 0
                elif dx < 0:
                    weight[:, :, dx:] = 0
                total += weight * neighbor
                mass += weight
            smooth = torch.where(
                mass > 1e-4,
                (total / mass.clamp_min(1e-6)).clamp_min(1e-6).reciprocal(),
                depth,
            )
            return {
                k: value[None]
                for k, value in dict(
                    depth=smooth,
                    prior=prior,
                    confidence=confidence,
                    margin=margin,
                    cost=cost,
                    improvement=improvement,
                    support=agreement,
                    disagreement=disagreement,
                    valid=valid,
                ).items()
            }

    @torch.no_grad()
    def independent_support(self, prepared, initial, required_sources):
        """Require reciprocal depth agreement from several distinct cameras.

        Use the selected, spatially filtered hypotheses rather than the old depth
        prior. A repeated texture matching in only one other view is insufficient.
        """
        depth = prepared["depth"][0].float()
        v, h, w = depth.shape
        E, K = initial.extrinsics[0].float(), initial.intrinsics[0].float()
        centers = torch.linalg.inv(E)[:, :3, 3]
        prior = gaussian_depth(initial)[0].float().clamp_min(1e-6)
        rays = (
            initial.gaussians.means[0].float().reshape(v, h, w, 3)
            - centers[:, None, None]
        ) / prior[..., None]
        xyz = centers[:, None, None] + rays * depth[..., None]
        agreements = []
        for i in range(v):
            votes = []
            for j in range(v):
                if i == j:
                    continue
                grid, z = project(xyz[i], E[j], K[j], h, w)
                reference = sample(depth[j, None], grid)[0]
                logerr = (z.clamp_min(1e-6) / reference.clamp_min(1e-6)).log().abs()
                inside = (grid.abs() < 0.985).all(-1) & (z > 0)
                inside &= sample(prepared["valid"][0, j, None], grid)[0] > 0.99
                votes.append(torch.exp(-logerr.square() / (2 * 0.04**2)) * inside)
            agreements.append(
                torch.stack(votes).topk(min(required_sources, v - 1), dim=0).values[-1]
            )
        return torch.stack(agreements)[None]

    def forward(self, rgb, initial, source_valid=None, prepared=None):
        p = self.prepare(rgb, initial, source_valid) if prepared is None else prepared
        prior = p["prior"]
        match = p["depth"]
        b, v, h, w = prior.shape
        if v > 3 and self.config["wide_consistency_min_sources"] > 1:
            p = dict(p)
            with torch.autocast(rgb.device.type, enabled=False):
                support = self.independent_support(
                    p, initial, self.config["wide_consistency_min_sources"]
                )
            p["confidence"] = p["confidence"] / p["support"].clamp_min(1e-6) * support
            p["support"] = support
            # Derived line evidence depends on the consistency rule.
            p.pop("line_depth", None)
            p.pop("line_weight", None)
        confidence = p["confidence"]
        if self.config["line_support"] and "line_depth" not in p:
            from .line_consensus import refine_lines

            line_depth, line_weight = refine_lines(
                rgb[0], match[0], confidence[0], p["support"][0], p["valid"][0]
            )
            p["line_depth"] = line_depth[None]
            p["line_weight"] = line_weight[None]
        if self.config["line_support"]:
            line_mask = p["line_weight"] > 0.0
            match = torch.where(line_mask, p["line_depth"], match)
            confidence = torch.where(
                line_mask, torch.maximum(confidence, p["line_weight"]), confidence
            )
        transfer_mask = torch.zeros_like(confidence, dtype=torch.bool)
        if self.config["source_transfer"]:
            from .depth_transfer import transfer_depth

            match, transfer_confidence = transfer_depth(
                rgb, initial, match, confidence, p["valid"]
            )
            transfer_mask = transfer_confidence > 0
            confidence = torch.maximum(confidence, transfer_confidence)
        rgb32 = rgb.float()
        gray = rgb32.mean(2)
        contrast = gray - F.avg_pool2d(gray.flatten(0, 1)[:, None], 9, 1, 4)[
            :, 0
        ].reshape_as(gray)
        ratio = (match / prior.clamp_min(1e-6)).log()
        fields = [
            ratio,
            confidence,
            p["margin"].clamp_max(0.2) * 5,
            p["cost"],
            p["improvement"],
            p["support"],
            p["disagreement"],
            contrast,
            p["valid"],
            prior.log() - prior.log().flatten(2).mean(-1)[:, :, None, None],
        ]
        inputs = torch.cat([rgb32, torch.stack(fields, 2)], 2).flatten(0, 1)
        delta = (
            self.decoder(inputs).reshape(b, v, 3, h, w)
            if self.config["learned"]
            else inputs.new_zeros(b, v, 3, h, w)
        )
        # Seed: accept independent geometric agreement rather than a prior-relative bound.
        base = (confidence * 2).clamp(0.001, 0.999)
        gate = torch.sigmoid(torch.logit(base) + delta[:, :, 0])
        supported = (
            (p["support"] > self.config["min_support"])
            & (p["improvement"] >= self.config["min_improvement"])
            & (p["margin"] >= self.config["min_margin"])
        )
        if self.config["line_support"]:
            supported = supported | (p["line_weight"] > 0)
        supported = supported | transfer_mask
        gate = gate * p["valid"] * supported
        if self.config["hard_gate"]:
            discrete = (gate > 0.5).float()
            gate = gate + (discrete - gate).detach() if self.training else discrete
        depth = prior * torch.exp(gate * (ratio + 0.05 * delta[:, :, 1].tanh()))
        # Evaluate visibility as a soft confidence, without deleting singly observed surfaces.
        opacity_gate = 1 - self.config["visibility_strength"] * p["disagreement"] * (
            1 - p["confidence"]
        )
        if self.config["learned"]:
            opacity_gate = opacity_gate * torch.exp(0.3 * delta[:, :, 2].tanh())
        with torch.autocast(rgb.device.type, enabled=False):
            z = gaussian_depth(initial).float().clamp_min(1e-6)
            centers = torch.linalg.inv(initial.extrinsics.float())[..., :3, 3][
                :, :, None, None
            ]
            means = initial.gaussians.means.float().reshape(b, v, h, w, 3)
            scale = (depth / z)[..., None]
            gs = replace(
                initial.gaussians,
                means=(centers + (means - centers) * scale).reshape(b, -1, 3),
                scales=(
                    initial.gaussians.scales.float().reshape(b, v, h, w, 3) * scale
                ).reshape(b, -1, 3),
                opacities=(
                    initial.gaussians.opacities.float().reshape(b, v, h, w)
                    * opacity_gate
                )
                .clamp(0, 1)
                .reshape_as(initial.gaussians.opacities),
            )
            if self.config["conflict_strength"] and v > 1:
                xyz = gs.means.reshape(b, v, h, w, 3)
                conflicts = []
                for i in range(v):
                    votes = []
                    for j in range(v):
                        if i == j:
                            continue
                        grid, cz = project(
                            xyz[0, i],
                            initial.extrinsics[0, j].float(),
                            initial.intrinsics[0, j].float(),
                            h,
                            w,
                        )
                        reference = sample(match[0, j, None], grid)[0]
                        reliability = sample(confidence[0, j, None], grid)[0]
                        color = sample(rgb32[0, j], grid)
                        mismatch = (color - rgb32[0, i]).abs().mean(0)
                        inside = (
                            (grid.abs() < 0.985).all(-1)
                            & (cz > 0)
                            & (sample(p["valid"][0, j, None], grid)[0] > 0.99)
                        )
                        contradiction = (
                            (cz < reference * 0.90) & (mismatch > 0.04) & inside
                        )
                        votes.append(contradiction * reliability)
                    evidence = (
                        torch.stack(votes).topk(min(2, v - 1), dim=0).values.mean(0)
                    )
                    conflicts.append(evidence)
                conflict = torch.stack(conflicts)[None]
                attenuation = torch.exp(-self.config["conflict_strength"] * conflict)
                gs = replace(
                    gs,
                    opacities=(
                        gs.opacities.reshape(b, v, h, w) * attenuation
                    ).reshape_as(gs.opacities),
                )
        self.last_audit = dict(
            gate=gate.detach(),
            matched_depth=match.detach(),
            corrected_depth=depth.detach(),
            confidence=p["confidence"],
        )
        return replace(
            initial,
            gaussians=gs,
            depth=depth,
            regularization=initial.regularization + 1e-4 * delta.square().mean(),
        )
