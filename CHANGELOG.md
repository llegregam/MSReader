# Changelog

## [1.7.0] - 2024-09-09

### Added

- Changes introduced in new versions are now explicitly detailed in the graphical user interface

### Changed

- Ratios are not calculated any more, just parsed from data

### Fixed

- Fixed bug on launch where the program wouldn't start because of multiple 
  calls to set_config
- Removed deprecated pandas functions

## [1.6.2] - 2024-03-27

### Added

- Rebuilt test suite to include new features
- Added testing support for Python 3.11 & 3.12

## [1.6.0] - 2024-03-20

### Changed

- Skyline data input format is now tabular with 'tsv' or 'txt' extension

### Added

- Added system to handle infinity signs which are now coerced to 'NaN'

[1.6.0]: https://github.com/llegregam/MSReader/releases/tag/v1.6.0
