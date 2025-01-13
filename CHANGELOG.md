# CHANGELOG


## v0.14.0 (2025-01-13)

### Bug Fixes

* fix: load weights to correct device and do not sort snake ([`e85d518`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e85d5181cc436de59a8886c385c0f938733f188c))

* fix: remove erosion and edge effects ([`33610fc`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/33610fca3c3fa04eeab0fa01098f4c6a2081d463))

* fix: specify RGB to avoid bug ([`d811287`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d811287690ab231bd092e536c4801c3020ee6dc2))

### Features

* feat: optional automatic estimation of num peaks ([`9f9deff`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9f9deff0b95abcbc86669cd82a063de8f74d0117))

### Unknown

* Merge pull request #16 from Ahus-AIM/fixes

Fixes ([`f7569f2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f7569f26809911d2cf2e971d2e6a70e2be61ac5d))


## v0.13.0 (2025-01-13)

### Bug Fixes

* fix: improve naming of in- & out channels for unet ([`44c3588`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/44c358854fe648a8933bd9d662e7ba611964f908))

* fix: improve control flow ([`1591018`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/15910184133ced1634d1844300a1dbc9dd2ea8d9))

* fix: flip metric comparison sign as we only use loss ([`09b563d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/09b563db57b24d9b7acc7c4b16cf5027ccf33c4d))

* fix: enforce uniform depth for all unet en-/decoder blocks ([`d2ab89e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d2ab89ea0edcba5eb6255f957852d447640aec65))

* fix: correct WeightedDiceLoss for batch size > 1 ([`2f94966`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2f94966935ac1b113e0fe21e9b3f3c173a70b517))

* fix: make hyperparameter search scheduler optional ([`10711a1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/10711a1386b592b9acd18355993981c10ee856b6))

* fix: seed hyperparameter search for uniform configuration each run ([`14b0a46`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/14b0a462710b280ad10a8119371f8f29e99debaa))

* fix: configure hyperparameter search with searchspace kwargs ([`8c27bd1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8c27bd15285b09a06d4af82c51482fffa4bb1b33))

### Features

* feat: implement cosine to constant lr scheduler ([`12c950d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/12c950db16c83dc4ce6bab12a4d4f8d6f0c5875b))

* feat: config file for unet hyperparameter search ([`c2236a6`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/c2236a682ea01ac6713919666b0c74471d05f10a))

* feat: support kwargs for metrics ([`5b1a4fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5b1a4fe039d09c44ba10a8f960151d0cc9960b4e))

* feat: enable early stopping of training ([`2eea233`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2eea23322d442a20854e8493c4a6deab3d1b9868))

* feat: add learning rate scheduler ([`e3bf78b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e3bf78bfe8484b5fba1605568824cd6f91fb645c))

* feat: implement MulticlassBinaryDiceLoss ([`d25e3d5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d25e3d589655c3c72a665fdc38b9a38df346f48e))

* feat: enable custom union exponent

This is suggested in https://arxiv.org/abs/1606.04797. ([`b1742e3`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b1742e37abe36935331e4fe2149af2918a2c5da7))

### Performance Improvements

* perf: configure unet hyperparameter search ([`845ece9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/845ece93ccdb404d8232d286a2b5d2fbac482aa2))

* perf: increase performance by pinning dataloader memory ([`6da1eca`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6da1eca25222a68f011897d5d7b8e9060680ab2b))

* perf: calculate running metrics due to memory limitations ([`75609c7`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/75609c7c44b75cf765f7d2e0eeefd6049f396447))

* perf: load data samples lazily due to memory constraints ([`7bd647a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7bd647a8b8706b53913e2b5433ccc7a0ac8092c8))

### Refactoring

* refactor: create class for binarization of multiclass loss ([`e1e1625`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e1e162553478ac56a19d2d35894777b4edc76d6b))

### Testing

* test: remove compilation as it fails and is currently not necessary ([`f7fb902`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f7fb9027973329eb02b702af37dad40c841c433a))

### Unknown

* Merge pull request #15 from Ahus-AIM/train_segmentation

Train segmentation ([`b7b2b54`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b7b2b547abbe8a5055dd00af2204a5609a7d97a5))


## v0.12.0 (2025-01-10)

### Bug Fixes

* fix: do not crash on bad segmentation input ([`827f10b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/827f10bc580251508457d157b557caf11bf02204))

### Features

* feat: add maximum limit on nonzero pixels ([`7c2daa7`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7c2daa7e9c8cbea925a1d2b1cefc8f6558972214))

* feat: digitization wrapper for inference ([`6be01c8`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6be01c8116ee83e941b8a60f6c2a9ba13a281480))

### Unknown

* Merge pull request #14 from Ahus-AIM/inference

Inference ([`afb6d59`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/afb6d59ff24082d9be07004dc0b314d3f801651e))


## v0.11.0 (2025-01-08)

### Features

* feat: better snake init and interpolation of missing values ([`3979227`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/397922770e6fcac6712ab0c60eded8c405e1fc46))

### Unknown

* Merge pull request #13 from Ahus-AIM/snake_interp

feat: better snake init and interpolation of missing values ([`db36e33`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/db36e33d217741666ea0cb1a5887cfb4beda1ebc))


## v0.10.0 (2025-01-08)

### Features

* feat: enable huggingface segmentation models

The SegFormer model is configured. ([`afc4b9a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/afc4b9a9d5be8cbb032bef174f9a546dfb247541))

### Unknown

* Merge pull request #12 from Ahus-AIM/huggingface_segmentation

feat: enable huggingface segmentation models ([`bae4a35`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/bae4a3599fd95e67008498c86e393c3c40e7cc95))


## v0.9.0 (2025-01-08)

### Features

* feat: perspective and updated random resize ([`a0b48cc`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a0b48cc9e14996cbce2c66c3d2eeb0a97a3b83cf))

* feat: cache transformed images to disc ([`37f8716`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/37f8716c664714d5cf94c7d6683e74d23a9ddda3))

### Unknown

* Merge pull request #11 from Ahus-AIM/improved_training_loop

Cache transforms ([`da8f686`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/da8f68660d46eaba9cfda31cd007114e654deb63))


## v0.8.1 (2025-01-06)

### Bug Fixes

* fix: reduce edge effects for robustness ([`96b69d1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/96b69d1bfb89f5c22adc2bf31e2bb1bfd2caabe6))

* fix: require grid pixels to be minima ([`1379dbe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1379dbea386b53d5c48d0621058569b5ad3f9dda))

### Documentation

* docs: example of perspective estimation on real image ([`b5b984d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b5b984d91c0e142c7f154bd6748e766783832b11))

* docs: pgf figure image to angle/radius ([`f8bb721`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f8bb721079d00b64b7ce08d780bd04f304baecdb))

* docs: illustrate perspective correction ([`098c0c9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/098c0c97083b1522fc016930babb8fb819055650))

### Unknown

* Merge pull request #10 from Ahus-AIM/perspective_figure

Perspective figure ([`5ddc2eb`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5ddc2eb8a2bb25e800b1b17512592ec50112cace))


## v0.8.0 (2024-12-27)

### Features

* feat: perspective detector handles binarization ([`2b743d2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2b743d28a4d2a85a8befb3885ebb9d1bbea59951))

### Unknown

* Merge pull request #9 from Ahus-AIM/binarize_in_perspective

Perspective RGB ([`9b69d6d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/9b69d6d30be3a590d15aec87078538946df6e337))


## v0.7.0 (2024-12-27)

### Bug Fixes

* fix: probabilities sum to 1 ([`8b477fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/8b477fe86cf9bf6fd9a73fead9377d566d645997))

### Features

* feat: add support for composed transforms ([`5b931bb`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5b931bbacbbb52a1f649eafa44e1edd9bed48f22))

* feat: add transforms and optional reduced load size ([`01d2a59`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/01d2a59b2c777297e13e52b8cff377c222305765))

### Unknown

* Merge pull request #8 from Ahus-AIM/improved_training_loop

Improved training loop ([`00a85ad`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/00a85ad3bb973aee90b4d73d7114209dd15f1b07))


## v0.6.0 (2024-12-23)

### Features

* feat: perspective detection for ecg paper ([`e99d2ad`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e99d2ad63237908593472003d12fde5aa8395a00))

### Testing

* test: perspective detector returns reasonable src points ([`49e4c32`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/49e4c32722fd957cee6400c783693f02f067c1ba))

### Unknown

* Merge pull request #7 from Ahus-AIM/perspective

Perspective detection ([`156abac`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/156abacdcb1093a004c1abf0c75bc49661a5632f))


## v0.5.0 (2024-12-20)

### Features

* feat: ecg scan transforms and corresponding visualization ([`94192fd`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/94192fd83795aba32895472249c2634bb1d5ee28))

### Unknown

* Merge pull request #5 from Ahus-AIM/transforms

Transformations and visualization ([`2024545`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2024545e9cc7e012cd68465b0cba807519d66ea0))


## v0.4.0 (2024-12-19)

### Bug Fixes

* fix: patch off by one epoch offset ([`a8e9713`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a8e971320c909de327ad2b9ef7b128b042563b77))

* fix: cast metrics output to numpy as tune does not support tensors ([`15c9819`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/15c98198ab12039c745e003858f1620c793b9a87))

* fix: correct input order of predictions and targets to metrics ([`bf62166`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/bf6216620612e5e9101660e5d81e1c01e8a934a7))

* fix: initialize metric classes ([`da3dc7e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/da3dc7e69f0a1b5f24764e8631f73c4b6f814566))

* fix: store raw torch tensors to from predictions and targets ([`a44930e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a44930e933ee0493b7d5d8f0869c4a28fa969b4f))

* fix: correct scaling of WeightedCrossEntropyLoss

There are two fixes, 1 is subtracted from alpha st. to match the
docstring description. Additionally, the loss is scaled st. the choice
of alpha does not significantly increase/decrease the total loss. ([`2b97ab1`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2b97ab172b2b7b9fec410d53898511a67ce45c85))

### Continuous Integration

* ci: increase max test time to 5 minutes ([`1611a5b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/1611a5be6f7e73fa6c2b75fda6934f85971727f7))

### Documentation

* docs: add marker for google docstring convention ([`f258d28`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f258d28d0a7577268122d3f5a264b571f3e002f6))

### Features

* feat: train dummy segmentation network ([`5cb3264`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5cb32645990dc9b4cf0a367c3a7f64211b2847d6))

* feat: implement MulticlassBinaryCrossEntropyLoss ([`30b7aea`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/30b7aea13105685609822af41a5448b5cfc8a5d6))

* feat: set up tensorboard ([`ecea315`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ecea3158821dc1cb45118a94017a7b1ac1fcf4bf))

* feat: split ecg dataset into train, val and test ([`6715f97`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/6715f977a16fe15afbd01df984eaf9973bfe6ed2))

### Performance Improvements

* perf: add flag for cudnn_benchmark ([`dc50b21`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/dc50b21734a3c0e697bcb086281ba11931d7ba8f))

* perf: compile train function if not using ray and mixed precision

Ray (and the tests) needs the functions to be picklable. Compilation
with mixed precision is not picklable. Training compilation and mixed
precision works fine without tests and ray. ([`0d7eb8d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/0d7eb8dac14c95c03764011de9f8e8cde0bb3cf3))

* perf: support mixed precision

Some type changes were also needed in order to fully support this
functionality. ([`7f10101`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7f10101a3239a5d7237ca92afd4754bca74da6dd))

### Unknown

* Merge pull request #6 from Ahus-AIM/train_unet_segmentation

Train unet segmentation ([`5316262`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/531626248d4a85e5685a244744505a8faae64430))


## v0.3.0 (2024-12-18)

### Bug Fixes

* fix: rename to enable __call__ ([`7047811`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/70478110c09a85dfc90d7a9630454ee02d576e2c))

* fix: remove torch.compile from inside model class ([`23ec2fe`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/23ec2febe7d51d55a7c3c727e03e2a1d0c64ea68))

* fix: remove superfluous argument ([`398d925`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/398d9254ed25722cc4f34321256f30c31f552c78))

### Documentation

* docs: add docstrings to grid detectors ([`097b6f4`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/097b6f4f4c36fd344036d1a884f50a4f291340d4))

### Features

* feat: multi-scale grid detection ([`7328f4e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7328f4edb59d0acfce8f82c40198643e5206ebd8))

### Testing

* test: make sure multiscale grid detector runs ([`ed3e9d0`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ed3e9d0f1afaee14cb9251cf8e5ceeaff053bc14))

### Unknown

* Merge pull request #4 from Ahus-AIM/adaptive_grid

Multi Scale Grid Detection ([`51b7194`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/51b71942af5dee56488285bce67ddacbe673f1fb))


## v0.2.0 (2024-12-16)

### Bug Fixes

* fix: one hot encode target mask ([`f7592b9`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f7592b96918105929d8d64b873889e759b7d60e7))

* fix: path for dice loss ([`3b7c3ba`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/3b7c3ba959ef4cee1d87f540263083c5676e7f60))

* fix: use custom loss ([`ccb697b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ccb697b8069d53007de44c44cbfdb211ba82fce5))

### Continuous Integration

* ci: ignore long comments ([`32d0c02`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/32d0c023e25ab801f9f6d0e1e43fd7ba37ca4b3e))

### Features

* feat: snake fitter, no cuda support yet ([`a0d659b`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a0d659bb7ba0853d3f06d292392ecd5d6dab4848))

* feat: grid detection with cuda support ([`456f152`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/456f152b6a848ccbbaefdfc7e18e3f170c55e704))

* feat: snake and segmentation losses ([`7be3a0a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/7be3a0a2ca9a6d55f2fcc7c827601fe8d3d0c182))

### Testing

* test: change model params for faster testing ([`888ab20`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/888ab20cd1056830456d98e0014e98dbd70cb070))

### Unknown

* Merge pull request #3 from Ahus-AIM/snakes_etc

Snakes, Grid detection, Losses ([`0c099df`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/0c099df5e4a3a2fcffe29629856423c405c3475c))


## v0.1.0 (2024-12-13)

### Continuous Integration

* ci: add .gitignore ([`b9a29b2`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/b9a29b2386f63cee7507de64fa890e782186087f))

* ci: typecheck python with mypy ([`70c6fdf`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/70c6fdf2b73fb5c1dd8edc1d46a261d070bfec3f))

* ci: ignore E203 for flake8 as black handles formatting ([`40659ee`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/40659eec5abcda2fb19761d750d3fb367200d54d))

* ci: use black in addition to flake8 for formatting ([`e3208b5`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e3208b51fe37ee2269a370c4ea12f15d3060514c))

* ci: increase flake8 line length to 120 ([`d0c522a`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/d0c522a4c00296fe6af49ed5d1adc17e9733c009))

### Documentation

* docs: add test status badge ([`ddd700e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/ddd700ea9cc0e3b65d949398070df6ce6a8b35cd))

### Features

* feat: set up project structure ([`a84603f`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/a84603f2f6745e01c4eb9171f782a2c7aa31786b))

### Testing

* test: set up pytest ([`2641fe3`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/2641fe32ceb1c6802d0782ba6d1f80828d32e7e7))

### Unknown

* Merge pull request #2 from Ahus-AIM/project_structure

Project structure ([`48d950d`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/48d950d08febc8455965eff7cf071c8f56ecddd8))


## v0.0.0 (2024-12-06)

### Chores

* chore: set up semantic versioning ([`4dda991`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/4dda9913d3c50108714993c2aecbc40cf407699c))

### Continuous Integration

* ci: lint python ([`f26d088`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/f26d0881fb049641f016d9f42acfd2572faf1f7b))

* ci: lint commits ([`e51f517`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e51f517fd72669ffb5eeb48736de9f3df73f574c))

### Documentation

* docs: initial commit ([`e16f1cb`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/e16f1cbd0fdfd7fa08a0687acd7f8091dc75cc69))

### Unknown

* Merge pull request #1 from Ahus-AIM/set_up_repository

Set up automatic linting and semantic versioning ([`5ccab2e`](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/commit/5ccab2e3577c5e5d28a8b84ccd3be11793ecd3fb))
