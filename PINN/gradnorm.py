class AdaptivePINN:   
    def update_weights(self, grad_phys, grad_bnd, grad_data):
        """Update adaptive loss weights based on gradient norms"""
        # Compute gradient norms
        def safe_norm(grad_list):
            norm_sq = 0.0
            for g in grad_list:
                if g is not None:
                    norm_sq += tf.reduce_sum(tf.square(g))
            return tf.sqrt(norm_sq + self.eps)

        n_phys = safe_norm(grad_phys)
        n_bnd = safe_norm(grad_bnd)
        n_data = safe_norm(grad_data)

        # Update weights using gradient balancing
        if n_phys > 0 and n_bnd > 0 and n_data > 0:
            g_mean = (n_phys + n_bnd + n_data) / 3.0

            # Simple inverse gradient norm weighting
            new_w_phys = tf.maximum(g_mean / n_phys, 1e-3)
            new_w_bnd = tf.maximum(g_mean / n_bnd, 1e-3)
            new_w_data = tf.maximum(g_mean / n_data, 1e-3)

            # Clip to reasonable range
            new_w_phys = tf.minimum(new_w_phys, 1e3)
            new_w_bnd = tf.minimum(new_w_bnd, 1e3)
            new_w_data = tf.minimum(new_w_data, 1e3)

            # Apply with momentum for stability
            momentum = 0.9
            self.w_phys.assign(momentum * self.w_phys + (1 - momentum) * new_w_phys)
            self.w_bnd.assign(momentum * self.w_bnd + (1 - momentum) * new_w_bnd)
            self.w_data.assign(momentum * self.w_data + (1 - momentum) * new_w_data)