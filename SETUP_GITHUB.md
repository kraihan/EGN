# Publishing this repository

One-time steps, in order. Delete this file once done.

## 1. Start clean

The existing `kraihan/EGN` repository was a fork of an SPD-network codebase with no LICENSE file.
Its history still contains that third-party code, so overwriting `main` is not enough — a permissive
licence on a repository whose history redistributes unlicensed code is not defensible.

**Create a new repository, or reset the history:**

```bash
# Option A (cleanest): new repo, e.g. kraihan/egn-lib, then archive the old one
# Option B: orphan branch in place
git checkout --orphan clean-main
git rm -rf .
# copy this directory's contents in, then:
git add -A
git commit -m "EGN 0.2.2: universal SPD-manifold classifier"
git branch -D main && git branch -m main
git push -f origin main
```

Archive the old fork with a README line pointing here. Do not silently delete it — a dead link from
a paper or an old notebook is worse than an archived repo.

## 2. Repository settings

- **About** → description: `Universal classifier on the SPD manifold — Riemannian deep learning for covariance data, in PyTorch`
- **About** → website: `https://pypi.org/project/egnlib/`
- **Topics**: `spd-manifold` `riemannian-geometry` `covariance` `deep-learning` `pytorch` `eeg`
  `brain-computer-interface` `machine-learning` `geometric-deep-learning` `classification`

Topics are the main reason a repository surfaces in GitHub search. They cost nothing and are the
highest-return five minutes here.

## 3. First release

```bash
git tag -a v0.2.2 -m "EGN 0.2.2"
git push origin v0.2.2
```

Then create a GitHub Release from the tag. The `publish.yml` workflow uploads to PyPI on release via
trusted publishing — configure it once at
<https://pypi.org/manage/project/egnlib/settings/publishing/> with owner `kraihan`, repository name,
workflow `publish.yml`, environment `pypi`. No API token needs to live in repository secrets.

## 4. Fill in the placeholders

- `CITATION.cff` — the ORCID field is a placeholder. Register at <https://orcid.org> if you have not.
- `PAPER.md` — leave as a stub until a decision is public. See the comment at the top of that file.
- `README.md` — the CI badge resolves once the workflow has run at least once on `main`.

## 5. Discoverability

`egnlib` currently returns nothing on Google because it has no inbound links. In rough order of
impact:

1. This repository, with topics set — GitHub pages rank far faster than PyPI pages.
2. Two or three Kaggle notebooks titled for what people search: "EEG motor imagery classification
   with Riemannian geometry", "SPD manifold neural network tutorial". Each importing `egnlib` by
   name and linking here.
3. A preprint, once review is complete. A citing paper outweighs everything above.
4. GitHub Pages docs at `kraihan.github.io/EGN` — worth doing, but after the three above. A docs
   site with no inbound links has the same problem the PyPI page has now.

Note that "EGN" already means E(n) Equivariant Graph Neural Networks in the literature, and that
work has thousands of stars. The repository description and topics are what disambiguate you, which
is why step 2 above matters more than it looks.
