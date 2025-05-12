# Changelog

## [1.7.5] - 2025-05-12

### Fixed

- Fixed bug when changing concentration unit after adding metadata

## [1.7.4] - 2025-02-20

### Fixed

- Fixed a bug concerning outputs and the exported table.


## [1.7.3] - 2025-02-20

### Fixed

- Fixed the problem of certain widgets being disabled when selecting tables to output. 


## [1.7.2] - 2024-12-04

### Fixed

- Fixed feature filtering problem in output files, now all features are present.
- Removed the 'LLOQ' and 'ULOQ' columns from specific tables in the output files.


## [1.7.1] - 2024-09-24

### Fixed

- Fixed bug where infinite and empty cells were present in "Table.xlsx" files. They are now replace with "NA".  


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
