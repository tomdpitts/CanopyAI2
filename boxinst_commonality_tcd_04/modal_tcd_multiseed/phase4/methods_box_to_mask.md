# Methods — Box-to-mask segmentation module

*(Draft methods subsection for the deployed box→mask module. Describes the module as
used to produce the reported masks; the abandoned contrastive prototype-repel term is
omitted. Math is written to transcribe directly to LaTeX.)*

---

## Box-to-mask segmentation

Given the detector's individual-tree-crown boxes, we convert each box to an instance
mask with a **training-free** appearance model: no mask annotations are ever used, and
the model is estimated from the training boxes and frozen backbone features alone. The
guiding assumption is *commonality* — the cells belonging to crowns within a box share a
common appearance that is distinct from the surrounding non-tree background — which lets
a per-box foreground/background contrast recover the crown shape without pixel labels.

The model operates on the same frozen DINOv3 feature grid as the detector (stride
$s$ px/cell). It is fit on the training tiles and applied to the test tiles at that same
stride, and is grid-agnostic — it stores the patch stride rather than a tile size, so the
prototypes transfer unchanged to tiles of any size at the same stride.

### Feature whitening

Let $x_c \in \mathbb{R}^{D}$ be the backbone feature at grid cell $c$. We fit a whitening
transform on the cells that fall inside a training box or in clear background: a mean
$\mu$, the top-$d$ PCA basis $U\in\mathbb{R}^{D\times d}$ ($d{=}128$), and per-component
scales $\sigma$. Every cell is whitened and $\ell_2$-normalised to the unit hypersphere,
$$\hat z_c = \frac{(x_c-\mu)^\top U/\sigma}{\lVert (x_c-\mu)^\top U/\sigma\rVert}\in\mathbb{S}^{d-1},$$
so all subsequent likelihoods are directional (cosine) on the sphere.

### Foreground and background appearance mixtures

Crown and background appearance are modelled by two von Mises–Fisher-like mixtures with a
shared concentration $\kappa$. The **background** mixture $\{(w_j,G_j)\}_{j=1}^{K_{bg}}$
is fit by spherical $k$-means on clear-background and box-adjacent "ring" cells, giving a
log-likelihood
$$\ell_{bg}(\hat z)=\operatorname{logsumexp}_j\big(\log w_j + \kappa\,\hat z^\top G_j\big).$$
The **foreground** prototypes $\{C_k\}_{k=1}^{K}$ are initialised by spherical $k$-means on
in-box cells (restricted to those atypical of the background) and refined by the EM below.
Prototypes whose responsibility share falls below a threshold are pruned, so $K$ adapts to
the data (auto-$K$).

### Size-conditioned spatial prior

Crowns occupy a roughly disk-shaped, central region of their box. We encode this with a
prior $\pi$ over a crown's *relative* position in its box. The box is divided into an
$N\times N$ grid of relative-position bins $(u,v)\in[0,1]^2$ ($N{=}8$), and — separately
for three crown-size terciles $s$ — we learn a per-bin, per-prototype foreground mass
$\pi_{s,k}(u,v)$, with total $\pi_s(u,v)=\sum_k \pi_{s,k}(u,v)$. The prior is normalised so
the mean in-box foreground mass equals $\pi/4$ (the area fraction of the inscribed disk)
and capped per bin. Conditioning on size lets small crowns (which nearly fill their box)
and large crowns (with more background in the corners) carry different priors.

### Per-box posterior (inference)

For a detection box $b$ we take the cells whose centres lie in $b$ (with a half-cell pad).
Each cell $c$ receives an **appearance log-ratio**
$$a_c=\underbrace{\operatorname{logsumexp}_k\!\Big(\log\tfrac{\pi_{s,k}(u_c,v_c)}{\pi_s(u_c,v_c)}+\kappa\,\hat z_c^\top C_k\Big)}_{\ell_{fg}(\hat z_c;\,u_c,v_c,s)}\;-\;\ell_{bg}(\hat z_c),$$
i.e. a spatially-weighted foreground mixture likelihood minus the background likelihood,
where $(u_c,v_c)$ is the cell's relative position in $b$ and $s$ its size tercile. Because
the frozen backbone injects a near-uniform context bias that inflates the absolute
foreground likelihood of *all* in-box cells, we take the appearance signal to be each
cell's score **relative to the box**, recentring it to zero mean over the box,
$$\bar a_c = a_c - \tfrac{1}{|b|}\textstyle\sum_{c'\in b} a_{c'}.$$
The foreground posterior fuses this within-box appearance contrast with the size-conditioned
spatial prior,
$$P(\text{fg}\mid c)=\sigma\big(\bar a_c + \operatorname{logit}\pi_s(u_c,v_c)\big),$$
and cells with $P(\text{fg}\mid c)\ge 0.5$ form the instance mask, clipped to $b$. The
recentring is essential: without it the posterior reduces to the spatial prior and the
mask degenerates to a box-shaped fill; with it, the within-box appearance competition
carves the crown from box background.

### Training-free EM

The foreground prototypes $\{C_k\}$ and the spatial prior $\pi$ are estimated jointly by
EM over the training boxes. The **E-step** assigns each in-box cell a foreground
responsibility from the posterior $\sigma\!\big(a_c + \operatorname{logit}\pi_s(u_c,v_c)\big)$
using the **absolute** appearance log-ratio $a_c$ — *without* the within-box recentring;
the **M-step** re-estimates each prototype as the $\ell_2$-renormalised,
responsibility-weighted mean of its assigned cells, and accumulates the spatial prior from
the responsibilities per size tercile and position bin (followed by the $\pi/4$
normalisation and cap). Fitting the prototypes on *absolute* appearance keeps them from
absorbing box-internal background (fitting on the recentred score instead degrades the
masker to a box-shaped fill); the within-box recentring $a_c\!\to\!\bar a_c$ is therefore
applied **only at inference**, where it turns the learned per-cell scores into the
within-box contrast that carves the crown. Closed-canopy cells (category *canopy*, non-ITC) are an
**ignore** label throughout — never a background negative nor a foreground positive — so
that tree-like canopy cannot corrupt the tree-vs-ground contrast. No polygon or mask label
is read at any stage: the entire module is supervised only by the boxes.
