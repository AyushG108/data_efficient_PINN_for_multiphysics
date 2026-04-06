class AdaptivePINN:
    def __init__(self, model, Re=10.0, Pr=0.71, Ra=1e4):
        self.model = model
        self.nu = 1/Re
        self.alpha = 1/(Re*Pr)
        self.Pr, self.Ra = Pr, Ra

        # Adaptive loss weights (initial values)
        self.w_phys = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.w_bnd = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.w_data = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        # Tracking for adaptive weighting
        self.loss_history = []
        self.phys_loss_history = []
        self.bnd_loss_history = []
        self.data_loss_history = []
        self.weight_history = []
        self.iteration = 0

        # Adaptive weighting parameters
        self.adapt_every = 50  # Update weights every N iterations
        self.warmup_iters = 200  # Fixed weights for first N iterations
        self.eps = 1e-8