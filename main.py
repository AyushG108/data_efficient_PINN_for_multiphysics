if __name__ == "__main__":
    print("Setting up thermo-fluid cavity problem with adaptive loss weighting...")

    # Generate training data
    n_interior = 2000  # Reduced for faster testing
    xy_f = np.random.rand(n_interior, 2).astype(np.float32)
    print(f"Generated {n_interior} interior points")

    n_boundary = 300
    xy_b = np.vstack([
        np.c_[np.random.rand(n_boundary), np.ones(n_boundary)],    # Top (y=1)
        np.c_[np.random.rand(n_boundary), np.zeros(n_boundary)],   # Bottom (y=0)
        np.c_[np.zeros(n_boundary), np.random.rand(n_boundary)],   # Left (x=0)
        np.c_[np.ones(n_boundary), np.random.rand(n_boundary)]     # Right (x=1)
    ]).astype(np.float32)
    print(f"Generated {len(xy_b)} boundary points")

    # Load sparse temperature data
    print("Loading sparse temperature data...")
    try:
        sparse = load_sparse_temp_csv("temparature_field_sparse_coarse_Ra_10000.csv")
        xy_d, theta_d = sparse[:, :2], sparse[:, 2:3]
        print(f"Loaded {len(xy_d)} sparse temperature measurements")
    except FileNotFoundError:
        print("Warning: CSV file not found. Creating synthetic data...")
        # Create synthetic temperature field (hot left, cold right)
        xy_d = np.random.rand(100, 2).astype(np.float32)
        theta_d = (1.0 - xy_d[:, 0:1])  # Linear temperature profile

    # Set boundary conditions
    # Streamfunction at boundaries (should be constant, set to 0)
    psi_b = np.zeros((len(xy_b), 1), np.float32)

    # Velocity boundary conditions (lid-driven cavity)
    uv_b = np.zeros((len(xy_b), 2), np.float32)
    uv_b[np.isclose(xy_b[:, 1], 1.0), 0] = 1.0  # Top wall: u=1, v=0

    # Temperature boundary conditions
    theta_b = np.full((len(xy_b), 1), np.nan, np.float32)
    theta_b[np.isclose(xy_b[:, 0], 0.0)] = 1.0  # Left wall hot
    theta_b[np.isclose(xy_b[:, 0], 1.0)] = 0.0  # Right wall cold
    # Top and bottom walls: Neumann (insulated) -> NaN

    print(f"Boundary conditions: {np.sum(~np.isnan(theta_b))} Dirichlet, "
          f"{np.sum(np.isnan(theta_b))} Neumann")

    # Create network and adaptive PINN
    print("\nBuilding network...")
    net = build_network(hidden=(32, 32, 32, 32))  # Smaller network for testing
    net(tf.zeros((1, 2)))  # Build network
    print(f"Network built with {net.count_params()} parameters")

    pinn = AdaptivePINN(net, Re=10.0, Pr=0.71, Ra=10000.0)

    # Prepare data tuple
    train_data = (xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_d)

    # Train the model
    try:
        pinn.train(train_data, epochs=15000, lr=1e-3, log_every=200)  # Reduced epochs for testing
    except Exception as e:
        print(f"Training error: {e}")
        print("Trying with lower learning rate...")
        pinn = AdaptivePINN(net, Re=10.0, Pr=0.71, Ra=10000.0)
        pinn.train(train_data, epochs=15000, lr=5e-4, log_every=200)

    # Save final model
    if pinn.loss_history:
        net.save_weights("pinn_thermo_lid_final.weights.h5")
        print("\nFinal model saved to 'pinn_thermo_lid_final.weights.h5'")

        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        ax = axes[0, 0]
        ax.semilogy(pinn.loss_history, label='Total Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Total Training Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Individual losses
        ax = axes[0, 1]
        ax.semilogy(pinn.phys_loss_history, label='Physics Loss', alpha=0.8)
        ax.semilogy(pinn.bnd_loss_history, label='Boundary Loss', alpha=0.8)
        ax.semilogy(pinn.data_loss_history, label='Data Loss', alpha=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Individual Loss Components')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Adaptive weights
        ax = axes[1, 0]
        w_phys = [w['w_phys'] for w in pinn.weight_history]
        w_bnd = [w['w_bnd'] for w in pinn.weight_history]
        w_data = [w['w_data'] for w in pinn.weight_history]

        ax.plot(w_phys, label='Physics Weight', linewidth=2)
        ax.plot(w_bnd, label='Boundary Weight', linewidth=2)
        ax.plot(w_data, label='Data Weight', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Loss Weights')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Loss ratios (to show balancing)
        ax = axes[1, 1]
        if len(w_phys) == len(pinn.phys_loss_history):
            phys_norm = np.array(pinn.phys_loss_history) * np.array(w_phys)
            bnd_norm = np.array(pinn.bnd_loss_history) * np.array(w_bnd)
            data_norm = np.array(pinn.data_loss_history) * np.array(w_data)

            ax.semilogy(phys_norm, label='Physics (weighted)', alpha=0.8)
            ax.semilogy(bnd_norm, label='Boundary (weighted)', alpha=0.8)
            ax.semilogy(data_norm, label='Data (weighted)', alpha=0.8)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Weighted Loss')
            ax.set_title('Weighted Loss Components')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print final statistics
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total iterations: {pinn.iteration}")
        print(f"Final loss: {pinn.loss_history[-1]:.3e}")
        print(f"Final physics loss: {pinn.phys_loss_history[-1]:.3e}")
        print(f"Final boundary loss: {pinn.bnd_loss_history[-1]:.3e}")
        print(f"Final data loss: {pinn.data_loss_history[-1]:.3e}")
        print(f"Final adaptive weights:")
        print(f"  Physics: {float(pinn.w_phys.numpy()):.4f}")
        print(f"  Boundary: {float(pinn.w_bnd.numpy()):.4f}")
        print(f"  Data: {float(pinn.w_data.numpy()):.4f}")
        print("="*60)

        # Generate and plot velocity and temperature fields
        X, Y, u_pred, v_pred, T_pred, psi_pred, p_pred = plot_velocity_temperature_fields(pinn, resolution=50)

        # Generate cross-section plots
        plot_cross_sections(pinn, resolution=100)

        # Additional analysis: Nusselt number calculation
        print("\nCalculating Nusselt number...")
        # Nusselt number at hot wall (x=0)
        y_nu = np.linspace(0, 1, 100)
        x_nu = np.zeros_like(y_nu)
        xy_nu = np.stack([x_nu, y_nu], axis=1).astype(np.float32)

        # Predict temperature at hot wall
        pred_nu = pinn.model.predict(xy_nu, batch_size=1024, verbose=0)
        T_wall = pred_nu[:, 2]

        # Compute temperature gradient at wall (dT/dx at x=0)
        # Use finite difference with small epsilon
        epsilon = 1e-3
        xy_nu_eps = np.stack([x_nu + epsilon, y_nu], axis=1).astype(np.float32)
        pred_nu_eps = pinn.model.predict(xy_nu_eps, batch_size=1024, verbose=0)
        T_wall_eps = pred_nu_eps[:, 2]

        dT_dx = (T_wall_eps - T_wall) / epsilon

        # Nusselt number: Nu = -dT/dx (since T_hot=1, T_cold=0)
        Nu_local = -dT_dx
        Nu_avg = np.mean(Nu_local)

        print(f"Average Nusselt number at hot wall: {Nu_avg:.4f}")

        # Plot Nusselt number distribution
        plt.figure(figsize=(10, 6))
        plt.plot(y_nu, Nu_local, 'b-', linewidth=2)
        plt.axhline(y=Nu_avg, color='r', linestyle='--', label=f'Average Nu = {Nu_avg:.4f}')
        plt.xlabel('y')
        plt.ylabel('Local Nusselt Number')
        plt.title('Nusselt Number Distribution at Hot Wall (x=0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('nusselt_number.png', dpi=150, bbox_inches='tight')
        plt.show()

    else:
        print("No training history available.")