# CHANGELOG


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
