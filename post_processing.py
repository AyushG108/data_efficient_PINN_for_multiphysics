def plot_velocity_temperature_fields(pinn, resolution=50):
    """Plot u, v, T fields on a grid"""
    print("\nGenerating velocity and temperature fields...")

    # Create grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Flatten grid for prediction
    xy_grid = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)

    # Predict on grid
    predictions = pinn.model.predict(xy_grid, batch_size=1024, verbose=0)
    psi_pred = predictions[:, 0].reshape(X.shape)
    p_pred = predictions[:, 1].reshape(X.shape)
    T_pred = predictions[:, 2].reshape(X.shape)

    # Compute velocity components from streamfunction
    # Use finite differences for visualization
    u_pred = np.zeros_like(X)
    v_pred = np.zeros_like(X)

    # Central differences for interior points
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            u_pred[i, j] = (psi_pred[i+1, j] - psi_pred[i-1, j]) / (2*dy)
            v_pred[i, j] = -(psi_pred[i, j+1] - psi_pred[i, j-1]) / (2*dx)

    # Boundary conditions
    u_pred[0, :] = 0  # Bottom wall
    u_pred[-1, :] = 1  # Top wall (lid-driven)
    u_pred[:, 0] = 0  # Left wall
    u_pred[:, -1] = 0  # Right wall

    v_pred[0, :] = 0  # Bottom wall
    v_pred[-1, :] = 0  # Top wall
    v_pred[:, 0] = 0  # Left wall
    v_pred[:, -1] = 0  # Right wall

    # Compute velocity magnitude
    vel_magnitude = np.sqrt(u_pred**2 + v_pred**2)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PINN Predictions for Thermo-Fluid Cavity (Ra=10000)', fontsize=16)

    # Plot temperature field
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, T_pred, levels=50, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Temperature (T) Field')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    # Plot streamfunction
    ax = axes[0, 1]
    contour = ax.contourf(X, Y, psi_pred, levels=50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamfunction (ψ)')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    # Plot velocity magnitude
    ax = axes[0, 2]
    contour = ax.contourf(X, Y, vel_magnitude, levels=50, cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Magnitude')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    # Plot u-velocity component
    ax = axes[1, 0]
    contour = ax.contourf(X, Y, u_pred, levels=50, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Horizontal Velocity (u)')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    # Plot v-velocity component
    ax = axes[1, 1]
    contour = ax.contourf(X, Y, v_pred, levels=50, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vertical Velocity (v)')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    # Plot pressure field
    ax = axes[1, 2]
    contour = ax.contourf(X, Y, p_pred, levels=50, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure (p) Field')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.savefig('velocity_temperature_fields.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Create vector plot of velocity
    plt.figure(figsize=(10, 8))

    # Subsample for clearer vector plot
    stride = max(1, resolution // 20)
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    u_sub = u_pred[::stride, ::stride]
    v_sub = v_pred[::stride, ::stride]

    # Background temperature contour
    plt.contourf(X, Y, T_pred, levels=20, alpha=0.6, cmap='jet')
    plt.colorbar(label='Temperature')

    # Velocity vectors
    plt.quiver(X_sub, Y_sub, u_sub, v_sub, color='black', scale=20, width=0.003)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity Vectors over Temperature Field')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('velocity_vectors.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Create streamlines plot
    plt.figure(figsize=(10, 8))

    # Streamlines over temperature field
    plt.contourf(X, Y, T_pred, levels=30, alpha=0.7, cmap='jet')
    plt.colorbar(label='Temperature')

    # Streamlines
    plt.streamplot(X, Y, u_pred, v_pred, color='white', linewidth=1, density=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamlines over Temperature Field')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('streamlines.png', dpi=150, bbox_inches='tight')
    plt.show()

    return X, Y, u_pred, v_pred, T_pred, psi_pred, p_pred


def plot_cross_sections(pinn, resolution=100):
    """Plot cross-sections of velocity and temperature"""
    print("Generating cross-section plots...")

    # Create vertical line at x=0.5
    x_mid = 0.5
    y_vert = np.linspace(0, 1, resolution)
    xy_vert = np.stack([np.full_like(y_vert, x_mid), y_vert], axis=1).astype(np.float32)

    # Create horizontal line at y=0.5
    y_mid = 0.5
    x_horiz = np.linspace(0, 1, resolution)
    xy_horiz = np.stack([x_horiz, np.full_like(x_horiz, y_mid)], axis=1).astype(np.float32)

    # Predict on lines
    pred_vert = pinn.model.predict(xy_vert, batch_size=1024, verbose=0)
    pred_horiz = pinn.model.predict(xy_horiz, batch_size=1024, verbose=0)

    psi_vert, p_vert, T_vert = pred_vert[:, 0], pred_vert[:, 1], pred_vert[:, 2]
    psi_horiz, p_horiz, T_horiz = pred_horiz[:, 0], pred_horiz[:, 1], pred_horiz[:, 2]

    # Compute velocities using finite differences
    u_vert = np.gradient(psi_vert, y_vert)
    v_vert = np.zeros_like(u_vert)  # v should be 0 at centerline

    u_horiz = np.zeros_like(x_horiz)  # u should be 0 at mid-height
    v_horiz = -np.gradient(psi_horiz, x_horiz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Vertical profiles at x=0.5
    ax = axes[0, 0]
    ax.plot(u_vert, y_vert, 'b-', linewidth=2, label='u velocity')
    ax.plot(v_vert, y_vert, 'r--', linewidth=2, label='v velocity')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('y')
    ax.set_title(f'Velocity Profiles at x={x_mid}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(T_vert, y_vert, 'g-', linewidth=2)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('y')
    ax.set_title(f'Temperature Profile at x={x_mid}')
    ax.grid(True, alpha=0.3)

    # Horizontal profiles at y=0.5
    ax = axes[0, 1]
    ax.plot(x_horiz, u_horiz, 'b-', linewidth=2, label='u velocity')
    ax.plot(x_horiz, v_horiz, 'r--', linewidth=2, label='v velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('Velocity')
    ax.set_title(f'Velocity Profiles at y={y_mid}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_horiz, T_horiz, 'g-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Temperature')
    ax.set_title(f'Temperature Profile at y={y_mid}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_sections.png', dpi=150, bbox_inches='tight')
    plt.show()