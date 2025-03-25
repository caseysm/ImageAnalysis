# ImageAnalysis Refactoring: Detailed Migration Plan

This document provides a comprehensive mapping of functions and classes from the original codebase to the new structure, including information about interface changes and new components.

## Table of Contents
1. [In_Situ_Functions.py Migration](#1-in_situ_functionspy-migration)
2. [Mapping_Functions.py Migration](#2-mapping_functionspy-migration)
3. [Album_Functions.py Migration](#3-album_functionspy-migration)
4. [Script Files Migration](#4-script-files-migration)
5. [New Components to Create](#5-new-components-to-create)
6. [Integration Components](#6-integration-components)
7. [Implementation Notes](#7-implementation-notes)

## 1. In_Situ_Functions.py Migration

### Original File: `In_Situ_Functions.py`

#### Moving to: `ImageAnalysis/utils/io.py`

| Original Function | New Function/Class | Interface Changes |
|-------------------|-------------------|-------------------|
| `InSitu.Import_ND2_by_Tile_and_Well()` | `ImageLoader.load_nd2_image()` | Now a method of `ImageLoader` class; takes self as first argument |
| `InSitu.Assemble_Data_From_ND2()` | `ImageLoader.load_cycles_for_tile()` | Now a method of `ImageLoader` class; more consistent error handling |

[... rest of the content from ChangesDocument.txt ...] 