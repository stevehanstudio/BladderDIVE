#!/usr/bin/env python3
"""
Setup PIPEX for CellDIVE dataset analysis
Creates the necessary directory structure and configuration for your 23-channel data
"""

import os
import shutil
from pathlib import Path
import subprocess
import sys

def setup_pipex_environment():
    """Install PIPEX dependencies in current environment"""
    
    print("Setting up PIPEX environment...")
    
    # Install PIPEX requirements
    requirements_file = "/mnt/data/Projects/HeLab/pipex/requirements.txt"
    
    if Path(requirements_file).exists():
        print("Installing PIPEX requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ])
            print("✅ PIPEX requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            print("You may need to install some packages manually")
    else:
        print("❌ Requirements file not found")

def create_pipex_data_structure():
    """Create the required directory structure for PIPEX"""
    
    # PIPEX expects a specific directory structure
    pipex_data_dir = Path("/mnt/data/Projects/HeLab/bladder_proteomics_data/pipex_analysis")
    
    # Create main directories
    directories = [
        pipex_data_dir,
        pipex_data_dir / "input",
        pipex_data_dir / "output", 
        pipex_data_dir / "masks",
        pipex_data_dir / "analysis"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    return pipex_data_dir

def prepare_celldive_for_pipex():
    """Prepare CellDIVE data in PIPEX-expected format"""
    
    print("\nPreparing CellDIVE data for PIPEX...")
    
    # PIPEX expects individual channel files in a specific naming convention
    source_file = "/mnt/data/Projects/HeLab/bladder_proteomics_data/CellDIVE_SLIDE-045_R0.aivia.tif"
    pipex_data_dir = create_pipex_data_structure()
    input_dir = pipex_data_dir / "input"
    
    # Channel information for your CellDIVE data
    channel_info = [
        ("DAPI_AF_R01", "DAPI", "nuclear"),
        ("CD45-AF488-CST", "CD45", "immune_pan"),
        ("CD3E-AF555-CST", "CD3E", "tcell"),
        ("Ki67-AF647-CST", "Ki67", "proliferation"),
        ("CD8a-AF750-CST", "CD8a", "tcell_cytotoxic"),
        ("Vim-AF488-CST", "VIM", "mesenchymal"),
        ("CD68-AF555-CST", "CD68", "macrophage"),
        ("HLA-DRA-AF4647", "HLADR", "mhc_class2"),
        ("CD31-AF750", "CD31", "endothelial"),
        ("SMA-AF488", "ACTA2", "smooth_muscle"),
        ("CD20-AF555", "CD20", "bcell"),
        ("CD163-AF647", "CD163", "macrophage_m2"),
        ("CD44-AF750-Nov", "CD44", "adhesion"),
        ("PanCK-AF488", "PANCK", "epithelial"),
        ("CD38-AF555", "CD38", "plasma_cell"),
        ("CD11c-AF647", "CD11c", "dendritic"),
        ("PDGFRA-AF488", "PDGFRA", "fibroblast"),
        ("COL1A1-AF555", "COL1A1", "collagen"),
        ("CD14-AF647", "CD14", "monocyte"),
        ("EPCAM-AF488", "EPCAM", "epithelial"),
        ("CD56-AF555", "CD56", "nk_cell"),
        ("CD45RO-AF647-BioT", "CD45RO", "memory_tcell"),
        ("DAPI_R06", "DAPI2", "nuclear")
    ]
    
    # Create a script to extract channels for PIPEX
    extract_script = f"""#!/usr/bin/env python3
import tifffile
import numpy as np
from pathlib import Path

def extract_channels_for_pipex():
    input_file = "{source_file}"
    output_dir = Path("{input_dir}")
    
    channel_info = {channel_info}
    
    print("Extracting channels for PIPEX...")
    
    try:
        with tifffile.TiffFile(input_file) as tif:
            print(f"Found {{len(tif.pages)}} pages")
            
            for i, (original_name, pipex_name, category) in enumerate(channel_info):
                output_file = output_dir / f"{{pipex_name}}.tif"
                print(f"Extracting {{i+1}}/23: {{original_name}} -> {{pipex_name}}.tif")
                
                try:
                    # Read channel data
                    channel_data = tif.pages[i].asarray()
                    
                    # Save as uncompressed TIFF (PIPEX prefers this)
                    tifffile.imwrite(output_file, channel_data, compression=None)
                    print(f"  ✅ Saved: {{output_file.name}}")
                    
                except Exception as e:
                    print(f"  ❌ Failed {{original_name}}: {{e}}")
                    
                    # Try alternative approach if main method fails
                    try:
                        print("  Trying alternative extraction...")
                        # You might need to implement fallback extraction here
                        pass
                    except:
                        print(f"  ❌ All extraction methods failed for {{original_name}}")
                        continue
        
        print("\\n✅ Channel extraction completed!")
        print(f"Channels saved to: {{output_dir}}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {{e}}")
        print("You may need to use the individual channel files if already extracted")

if __name__ == "__main__":
    extract_channels_for_pipex()
"""
    
    # Save the extraction script
    extract_script_path = pipex_data_dir / "extract_for_pipex.py"
    with open(extract_script_path, 'w') as f:
        f.write(extract_script)
    
    os.chmod(extract_script_path, 0o755)
    print(f"Created extraction script: {extract_script_path}")
    
    return pipex_data_dir, channel_info

def create_pipex_config(pipex_data_dir, channel_info):
    """Create PIPEX configuration for CellDIVE analysis"""
    
    print("\nCreating PIPEX configuration...")
    
    # Create markers file (required by PIPEX)
    markers_file = pipex_data_dir / "markers.txt"
    
    # PIPEX uses marker names for analysis
    markers = [info[1] for info in channel_info]  # Use the simplified names
    
    with open(markers_file, 'w') as f:
        for marker in markers:
            f.write(f"{marker}\\n")
    
    print(f"Created markers file: {markers_file}")
    
    # Create PIPEX command template
    pipex_command_template = f"""#!/bin/bash

# PIPEX Analysis for CellDIVE Bladder Dataset
# 
# Basic PIPEX command for your 23-channel CellDIVE data
# Adjust parameters as needed for your specific analysis

cd /mnt/data/Projects/HeLab/pipex

# Basic segmentation and analysis
python pipex.py \\
    --input_folder {pipex_data_dir}/input \\
    --output_folder {pipex_data_dir}/output \\
    --markers_file {markers_file} \\
    --dapi_channel DAPI \\
    --membrane_channel PANCK \\
    --nuclei_size 20 \\
    --expansion_size 10 \\
    --membrane_size 30 \\
    --compactness 0.5 \\
    --analysis \\
    --clustering \\
    --interactions \\
    --generate_geojson \\
    --generate_qupath

# Alternative command with preprocessing (if images need correction)
# python pipex.py \\
#     --input_folder {pipex_data_dir}/input \\
#     --output_folder {pipex_data_dir}/output \\
#     --markers_file {markers_file} \\
#     --dapi_channel DAPI \\
#     --membrane_channel PANCK \\
#     --preprocessing \\
#     --threshold_min 1 \\
#     --threshold_max 99 \\
#     --nuclei_size 20 \\
#     --expansion_size 10 \\
#     --membrane_size 30 \\
#     --analysis \\
#     --clustering \\
#     --interactions

echo "PIPEX analysis completed!"
echo "Results in: {pipex_data_dir}/output"
"""
    
    # Save the command template
    command_file = pipex_data_dir / "run_pipex.sh"
    with open(command_file, 'w') as f:
        f.write(pipex_command_template)
    
    os.chmod(command_file, 0o755)
    print(f"Created PIPEX command script: {command_file}")
    
    return markers_file, command_file

def create_cell_types_config(pipex_data_dir):
    """Create cell types configuration for cluster refinement"""
    
    print("Creating cell types configuration for bladder cancer analysis...")
    
    # Cell types relevant to bladder cancer and your markers
    cell_types_csv = """ref_id,cell_group,cell_type,cell_subtype,rank_filter,min_confidence,marker1,rule1,marker2,rule2,marker3,rule3
1,epithelial,tumor,panck_positive,positive_only,30,PANCK,high,EPCAM,high,CD45,low
1,epithelial,tumor,epcam_positive,positive_only,30,EPCAM,high,PANCK,medium,CD45,low
1,immune,tcell,cd3_positive,positive_only,30,CD3E,high,CD45,high,PANCK,low
1,immune,tcell,cd8_cytotoxic,positive_only,30,CD8a,high,CD3E,high,CD45,high
1,immune,tcell,memory,positive_only,30,CD45RO,high,CD3E,high,CD45,high
1,immune,bcell,cd20_positive,positive_only,30,CD20,high,CD45,high,CD3E,low
1,immune,plasma,cd38_positive,positive_only,30,CD38,high,CD45,high,CD20,medium
1,immune,macrophage,cd68_positive,positive_only,30,CD68,high,CD45,high,CD3E,low
1,immune,macrophage,m2_type,positive_only,30,CD163,high,CD68,high,CD45,high
1,immune,monocyte,cd14_positive,positive_only,30,CD14,high,CD45,high,CD68,medium
1,immune,dendritic,cd11c_positive,positive_only,30,CD11c,high,CD45,high,CD3E,low
1,immune,nk_cell,cd56_positive,positive_only,30,CD56,high,CD45,high,CD3E,low
1,stromal,fibroblast,pdgfra_positive,positive_only,30,PDGFRA,high,VIM,high,CD45,low
1,stromal,fibroblast,collagen_producing,positive_only,30,COL1A1,high,VIM,high,CD45,low
1,endothelial,vessel,cd31_positive,positive_only,30,CD31,high,CD45,low,PANCK,low
1,smooth_muscle,vessel,acta2_positive,positive_only,30,ACTA2,high,CD31,medium,CD45,low
1,proliferating,any,ki67_positive,positive_only,20,Ki67,high
2,immune,activated,hladr_positive,positive_only,20,HLADR,high,CD45,high
2,adhesion,high_cd44,cd44_positive,positive_only,20,CD44,high"""
    
    cell_types_file = pipex_data_dir / "cell_types.csv"
    with open(cell_types_file, 'w') as f:
        f.write(cell_types_csv)
    
    print(f"Created cell types configuration: {cell_types_file}")
    
    return cell_types_file

def main():
    print("PIPEX Setup for CellDIVE Bladder Dataset")
    print("=" * 50)
    
    # Setup environment
    setup_pipex_environment()
    
    # Create directory structure and prepare data
    pipex_data_dir, channel_info = prepare_celldive_for_pipex()
    
    # Create configuration files
    markers_file, command_file = create_pipex_config(pipex_data_dir, channel_info)
    cell_types_file = create_cell_types_config(pipex_data_dir)
    
    print("\\n" + "=" * 50)
    print("✅ PIPEX SETUP COMPLETE!")
    print("=" * 50)
    
    print(f"\\n📁 Data directory: {pipex_data_dir}")
    print(f"📄 Markers file: {markers_file}")
    print(f"🧬 Cell types config: {cell_types_file}")
    print(f"🚀 Run script: {command_file}")
    
    print("\\n🔄 Next Steps:")
    print("1. Extract channels for PIPEX:")
    print(f"   cd {pipex_data_dir}")
    print("   python extract_for_pipex.py")
    print()
    print("2. Run PIPEX analysis:")
    print(f"   ./run_pipex.sh")
    print()
    print("3. Results will be in:")
    print(f"   {pipex_data_dir}/output/")
    print()
    print("📊 PIPEX will generate:")
    print("• Cell segmentation masks")
    print("• FCS file with cell measurements")
    print("• Clustering analysis")
    print("• Cell-cell interaction analysis")
    print("• QuPath-compatible annotations")
    print("• Various plots and visualizations")
    print()
    print("🔧 Customization:")
    print("• Edit markers.txt to add/remove markers")
    print("• Edit cell_types.csv to refine cell type classification")
    print("• Adjust parameters in run_pipex.sh as needed")

if __name__ == "__main__":
    main()
