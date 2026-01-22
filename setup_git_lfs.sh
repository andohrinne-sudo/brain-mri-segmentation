#!/bin/bash
echo "Installing Git LFS..."
git lfs install
echo "Configuring Git LFS to track .hdf5 and .h5 files..."
git lfs track "*.hdf5"
git lfs track "*.h5"
git add .gitattributes
echo "Force-adding model_pretrained.hdf5 to Git LFS..."
git add --force model_pretrained.hdf5
echo "Git LFS setup complete."
