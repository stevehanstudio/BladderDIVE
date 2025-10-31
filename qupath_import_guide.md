# QuPath Multi-Channel Import Guide for CellDIVE Data

## Problem
QuPath 0.6.0 not showing all 23 channels from CellDIVE_SLIDE-045_R0.aivia.tif

## Channel List (for reference)
Your file contains these 23 channels:
1. DAPI_AF_R01 (Nuclear)
2. CD45-AF488-CST (Pan-leukocyte)
3. CD3E-AF555-CST (T cells)
4. Ki67-AF647-CST (Proliferation)
5. CD8a-AF750-CST (Cytotoxic T cells)
6. Vim-AF488-CST (Vimentin)
7. CD68-AF555-CST (Macrophages)
8. HLA-DRA-AF4647 (MHC Class II)
9. CD31-AF750 (Endothelial)
10. SMA-AF488 (Smooth muscle)
11. CD20-AF555 (B cells)
12. CD163-AF647 (M2 Macrophages)
13. CD44-AF750-Nov (Cell adhesion)
14. PanCK-AF488 (Pan-cytokeratin)
15. CD38-AF555 (Plasma cells)
16. CD11c-AF647 (Dendritic cells)
17. PDGFRA-AF488 (Fibroblasts)
18. COL1A1-AF555 (Collagen)
19. CD14-AF647 (Monocytes)
20. EPCAM-AF488 (Epithelial)
21. CD56-AF555 (NK cells)
22. CD45RO-AF647-BioT (Memory T cells)
23. DAPI_R06 (Nuclear - second round)

## Solutions to Try

### Solution 1: Import with Bio-Formats
1. In QuPath, go to **File → Import → Images from disk**
2. Select your .tif file
3. In the import dialog, ensure "Use Bio-Formats" is checked
4. Try checking "Separate reader per core" if available
5. Look for channel settings and ensure all 23 are enabled

### Solution 2: Manual Channel Selection
1. After opening the image, go to **View → Brightness/Contrast**
2. Check if all 23 channels are listed in the channel selector
3. You may need to enable them individually

### Solution 3: Check Image Server Settings
1. In QuPath, go to **Edit → Preferences**
2. Look for "Image Servers" or "Bio-Formats" settings
3. Ensure multi-channel support is enabled

### Solution 4: Use Image Type Setting
1. After opening, go to **Image → Image type**
2. Try setting to "Immunofluorescence"
3. This may help QuPath recognize the multi-channel nature
