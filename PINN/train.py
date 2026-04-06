class AdaptivePINN:
    
    def train_step(self, xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data, optimizer):
        """Single training step"""
        self.iteration += 1

        # Compute losses and gradients
        L_phys, L_bnd, L_data, grad_phys, grad_bnd, grad_data = self.compute_losses(
            xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data
        )

        # Update adaptive weights if needed
        if self.iteration % self.adapt_every == 0 and self.iteration >= self.warmup_iters:
            self.update_weights(grad_phys, grad_bnd, grad_data)

        # Compute total loss with current weights
        total_loss = (self.w_phys * L_phys +
                     self.w_bnd * L_bnd +
                     self.w_data * L_data)

        # Compute gradients for total loss
        with tf.GradientTape() as tape:
            # We need to recompute the losses in this tape for gradient computation
            L_phys_re, L_bnd_re, L_data_re, _, _, _ = self.compute_losses(
                xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data
            )
            total_loss_re = (self.w_phys * L_phys_re +
                           self.w_bnd * L_bnd_re +
                           self.w_data * L_data_re)

        # Get gradients and update model
        grads = tape.gradient(total_loss_re, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Store history
        self.loss_history.append(float(total_loss))
        self.phys_loss_history.append(float(L_phys))
        self.bnd_loss_history.append(float(L_bnd))
        self.data_loss_history.append(float(L_data))
        self.weight_history.append({
            'w_phys': float(self.w_phys.numpy()),
            'w_bnd': float(self.w_bnd.numpy()),
            'w_data': float(self.w_data.numpy())
        })

        return total_loss

    def train(self, data, epochs=15000, lr=1e-3, log_every=100):
        """Main training loop with Adam optimizer"""
        # Unpack data
        xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data = data

        # Create Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Initial weights: w_phys={float(self.w_phys.numpy()):.3f}, "
              f"w_bnd={float(self.w_bnd.numpy()):.3f}, "
              f"w_data={float(self.w_data.numpy()):.3f}")
        print("-" * 80)

        best_loss = float('inf')

        start_time = datetime.now()

        for epoch in range(15000):
            # Training step
            loss = self.train_step(xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data, optimizer)

            # Logging
            if epoch % log_every == 0 or epoch == epochs - 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {float(loss):.3e} | "
                      f"Phys: {self.phys_loss_history[-1]:.3e} | "
                      f"Bnd: {self.bnd_loss_history[-1]:.3e} | "
                      f"Data: {self.data_loss_history[-1]:.3e} | "
                      f"Weights: [{float(self.w_phys.numpy()):.2e},"
                      f"{float(self.w_bnd.numpy()):.2e},"
                      f"{float(self.w_data.numpy()):.2e}] | "
                      f"Time: {elapsed:.1f}s")


        print(f"\nTraining completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
        print(f"Best loss: {best_loss:.3e}")

        # Load best model
        try:
            self.model.load_weights("best_model.weights.h5")
        except:
            pass