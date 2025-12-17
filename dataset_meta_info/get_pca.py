import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Any, Dict, List

class HandPosePCA:
    """
    A wrapper for scikit-learn's PCA to handle finger joint sequence data.
    
    This class handles the necessary reshaping of data from (frames, joints, dims)
    to a 2D matrix for PCA and back.
    """
    def __init__(self, n_components):
        """
        Args:
            n_components (int or float): The number of principal components to keep.
                - If int, it's the absolute number of components.
                - If float (e.g., 0.95), it's the amount of variance to explain.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.original_shape_info = {}

    def fit(self, data):
        """
        Fit the PCA model to the hand pose data.
        
        Args:
            data (np.ndarray): A single sequence of shape (frames, joints)
                               or a batch of sequences of shape (..., frames, joints).
        """
        # --- 1. Reshape Data for PCA ---
        # Store original shape for inverse transform
        self.original_shape_info = {
            'joints': data.shape[-1]
        }
        
        # --- 2. Fit the PCA model ---
        self.pca.fit(data)

        print(f"PCA model fitted.")
        print(f"Explained variance by {self.pca.n_components_} components: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def transform(self, data):
        """
        Transform pose data into the lower-dimensional latent space.
        
        Args:
            data (np.ndarray): Pose data with the same (joints, dims) as the fitted data.
        
        Returns:
            np.ndarray: The latent representation of shape (..., n_components).
        """
        original_shape = data.shape
        reshaped_data = data.reshape(1, -1) 
        # Project data onto principal components
        latent_representation = self.pca.transform(reshaped_data)
        
        # Reshape back to include sequence/frame structure
        output_shape = original_shape[:-1] + (self.pca.n_components_,)
        return latent_representation.reshape(output_shape)

    def inverse_transform(self, latent_data):
        """
        Reconstruct the full pose data from the latent representation.
        
        Args:
            latent_data (np.ndarray): Latent data of shape (..., n_components).
        
        Returns:
            np.ndarray: The reconstructed pose data in its original shape.
        """
        original_shape = latent_data.shape
        
        # Flatten for scikit-learn's inverse_transform
        reshaped_latent = latent_data.reshape(-1, self.pca.n_components_)

        # Reconstruct from latent space
        reconstructed_flat = self.pca.inverse_transform(reshaped_latent)
        
        # Reshape back to the original (frames, joints, dims) format
        output_shape = original_shape[:-1] + (self.original_shape_info['joints'],)
        return reconstructed_flat.reshape(output_shape)

    def plot_explained_variance(self):
        """Plots the cumulative explained variance."""
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Principal Components')
        plt.grid(True)
        # Add a line for 95% variance
        plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
        plt.legend()
        plt.savefig('dataset_meta_info/explained_variance.png')

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    from argparse import ArgumentParser
    import glob
    from pathlib import Path
    import json

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/orca_D4/annotation/train',
                        help='Path to the directory containing hand pose sequences.')
    args = parser.parse_args()
    # --- 1. Get the Data ---
    json_dir = Path(args.data_dir)
    if not json_dir.exists():
        print(f"Error: Directory '{args.data_dir}' does not exist")
        exit(0)
    
    # Find all JSON files
    json_files = sorted(glob.glob(str(json_dir / "*.json")))
    
    if not json_files:
        print(f"Error: No JSON files found in '{args.data_dir}'")
        exit(0)
    
    print(f"Found {len(json_files)} JSON files in {args.data_dir}")

    # Load all data for aggregate statistics
    print("\n" + "="*80)
    print("Loading all JSON files for aggregate statistics...")
    print("="*80)
    all_data = []
    for i, json_file in enumerate(json_files):
        try:
            data = load_json_data(json_file)
            joint_data = np.array(data['observation.state.hand_joint_position'])  # shape: (frames, joints, dims)
            all_data.append(joint_data)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(json_files)} files...")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")
    
    print(f"Successfully loaded {len(all_data)} JSON files")
    joint_data = np.concatenate(all_data, axis=0)  # shape: (total_frames, joints)
    print(f"Combined joint data shape: {joint_data.shape}")

    # --- 2. Initialize and Fit PCA Model ---
    # We want to find the number of components that explain 95% of the variance
    n_components = 0.95
    
    # You can use either class, the interface is the same
    # pca_model = HandPosePCANumpy(n_components=5) # use this line for the numpy version with 5 components
    pca_model = HandPosePCA(n_components=n_components)
    
    # Fit on all available data
    pca_model.fit(joint_data)

    # Plot to see how many components are needed
    if isinstance(pca_model, HandPosePCA):
        pca_model.plot_explained_variance()

    # --- 3. Transform and Reconstruct ---
    # Take one sequence to test
    test_sequence = joint_data[0]
    print(f"\nOriginal test sequence shape: {test_sequence.shape}")

    # Transform to latent space
    latent_representation = pca_model.transform(test_sequence)
    print(f"Transformed to latent space shape: {latent_representation.shape}")

    # Reconstruct from latent space
    reconstructed_sequence = pca_model.inverse_transform(latent_representation)
    print(f"Reconstructed sequence shape: {reconstructed_sequence.shape}")

    # --- 4. Evaluate Reconstruction Quality ---
    mse = np.mean((test_sequence - reconstructed_sequence) ** 2)
    print(f"\nMean Squared Error (MSE) of reconstruction: {mse:.6f}")

    # The principal components themselves represent "eigenposes" or primary modes of motion.
    # For the sklearn wrapper, they are stored in pca_model.pca.components_
    # For the numpy version, they are in pca_model.components
    # Each component has a shape of (N_JOINTS * N_DIMS)
    if isinstance(pca_model, HandPosePCA):
        first_principal_component_flat = pca_model.pca.components_[0]
        # You can reshape it to visualize it as a pose displacement
        print(f"\nShape of the first principal component (eigen-pose): {first_principal_component_flat.shape}")