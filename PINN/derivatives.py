class AdaptivePINN:
    def compute_derivatives(self, xy):
        """Compute all derivatives needed for physics losses"""
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            out = self.model(tf.concat([x, y], axis=1))
            psi, p, T = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            # First derivatives
            psi_x = tape.gradient(psi, x)
            psi_y = tape.gradient(psi, y)

            # Velocity components
            u = psi_y
            v = -psi_x

            # Velocity gradients
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)

            # Second derivatives
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)

            # Pressure gradients
            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)

            # Temperature gradients
            T_x = tape.gradient(T, x)
            T_y = tape.gradient(T, y)
            T_xx = tape.gradient(T_x, x)
            T_yy = tape.gradient(T_y, y)

        del tape

        return {
            'u': u, 'v': v,
            'u_x': u_x, 'u_y': u_y, 'v_x': v_x, 'v_y': v_y,
            'u_xx': u_xx, 'u_yy': u_yy, 'v_xx': v_xx, 'v_yy': v_yy,
            'p_x': p_x, 'p_y': p_y,
            'T': T, 'T_x': T_x, 'T_y': T_y, 'T_xx': T_xx, 'T_yy': T_yy
        }