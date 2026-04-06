
class AdaptivePINN:
    
    def compute_losses(self, xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data):
        """Compute all loss components"""
        # Convert inputs to tensors
        xy_f = tf.convert_to_tensor(xy_f, dtype=tf.float32)
        xy_b = tf.convert_to_tensor(xy_b, dtype=tf.float32)
        xy_d = tf.convert_to_tensor(xy_d, dtype=tf.float32)
        psi_b = tf.convert_to_tensor(psi_b, dtype=tf.float32)
        uv_b = tf.convert_to_tensor(uv_b, dtype=tf.float32)
        theta_b = tf.convert_to_tensor(theta_b, dtype=tf.float32)
        theta_data = tf.convert_to_tensor(theta_data, dtype=tf.float32)

        # Physics losses (interior points)
        with tf.GradientTape(persistent=True) as tape_phys:
            tape_phys.watch(xy_f)
            g = self.compute_derivatives(xy_f)

            # Physics residuals
            buoy = (self.Pr * self.Ra / 100.0) * g['T']
            res_x = g['u']*g['u_x'] + g['v']*g['u_y'] + g['p_x'] - self.nu*(g['u_xx'] + g['u_yy'])
            res_y = g['u']*g['v_x'] + g['v']*g['v_y'] + g['p_y'] - self.nu*(g['v_xx'] + g['v_yy']) + buoy
            res_T = g['u']*g['T_x'] + g['v']*g['T_y'] - self.alpha*(g['T_xx'] + g['T_yy'])

            L_phys = tf.reduce_mean(res_x**2 + res_y**2 + res_T**2)

        # Boundary losses
        with tf.GradientTape(persistent=True) as tape_bnd:
            tape_bnd.watch(xy_b)
            out_b = self.model(xy_b)
            psi_b_pred, _, T_b_pred = out_b[:, 0:1], out_b[:, 1:2], out_b[:, 2:3]

            # Velocity from streamfunction
            psi_x_b = tape_bnd.gradient(psi_b_pred, xy_b)[:, 0:1]
            psi_y_b = tape_bnd.gradient(psi_b_pred, xy_b)[:, 1:2]
            u_b_pred, v_b_pred = psi_y_b, -psi_x_b

            # Temperature gradient for Neumann BC
            T_y_b = tape_bnd.gradient(T_b_pred, xy_b)[:, 1:2]

            # Boundary loss components
            L_psi = tf.reduce_mean((psi_b_pred - psi_b)**2)
            L_uv = tf.reduce_mean((tf.concat([u_b_pred, v_b_pred], axis=1) - uv_b)**2)

            # Handle Dirichlet/Neumann BCs for temperature
            mask = tf.cast(~tf.math.is_nan(theta_b), tf.float32)
            tb = tf.where(tf.math.is_nan(theta_b), 0.0, theta_b)
            L_theta_dir = tf.reduce_mean(((T_b_pred - tb) * mask)**2)
            L_theta_neu = tf.reduce_mean((T_y_b * (1 - mask))**2)

            L_bnd = L_psi + L_uv + L_theta_dir + L_theta_neu

        # Data loss
        out_d = self.model(xy_d)
        T_data_pred = out_d[:, 2:3]
        L_data = tf.reduce_mean((T_data_pred - theta_data)**2)

        # Compute gradients for adaptive weighting
        grad_phys = tape_phys.gradient(L_phys, self.model.trainable_variables)
        grad_bnd = tape_bnd.gradient(L_bnd, self.model.trainable_variables)
        grad_data = tape_bnd.gradient(L_data, self.model.trainable_variables)

        del tape_phys, tape_bnd

        return L_phys, L_bnd, L_data, grad_phys, grad_bnd, grad_data

    def compute_total_loss(self, xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data):
        """Compute total loss with current weights"""
        L_phys, L_bnd, L_data, _, _, _ = self.compute_losses(
            xy_f, xy_b, xy_d, psi_b, uv_b, theta_b, theta_data
        )

        total_loss = (self.w_phys * L_phys +
                     self.w_bnd * L_bnd +
                     self.w_data * L_data)

        return total_loss, L_phys, L_bnd, L_data