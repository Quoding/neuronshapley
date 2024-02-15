import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.datasets import load_iris


class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 10, bias=False),
            nn.ReLU(),
            nn.Linear(10, 10, bias=False),
            nn.ReLU(),
            nn.Linear(10, out_size, bias=False),
        )

    def forward(self, x):
        return self.model(x)


class NetWrapper:
    def __init__(self, seed):
        self.seed = seed

        test_idx = torch.load(f"data/models/test_idx_{seed}.pth")
        X, y = load_iris(return_X_y=True)
        self.X, self.labels = X[test_idx], y[test_idx]

        self.init_from_torch()

    def init_from_torch(self):
        torch_model = Net(4, 3)
        torch_model.load_state_dict(torch.load(f"data/models/model_{self.seed}.pth"))

        layer_params = {}
        for name, param in torch_model.named_parameters():
            layer_params[name] = param.detach().numpy()

        self.input = tf.placeholder(
            tf.float32, shape=(None, self.X.shape[1]), name="input"
        )

        # First layer
        # hid1_size = 10
        w1 = tf.Variable(layer_params["model.0.weight"], name="input_layer")
        y1 = tf.nn.relu(tf.matmul(w1, tf.transpose(self.input)))

        # Second layer
        # hid2_size = 10
        w2 = tf.Variable(layer_params["model.2.weight"], name="layer_1")
        y2 = tf.nn.relu(tf.matmul(w2, y1))

        # Output layer
        wo = tf.Variable(layer_params["model.4.weight"], name="layer_2")
        self.logits = tf.transpose(tf.matmul(wo, y2))

        self.y_input = tf.constant(self.labels)

        self.ends = {}
        self.ends["logits"] = self.logits
        self.ends["input"] = self.input

        self.sample_accuracy = tf.cast(
            tf.math.equal(tf.math.argmax(self.ends["logits"], -1), self.y_input),
            tf.float32,
        )

        self.accuracy = tf.reduce_mean(self.sample_accuracy)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, f"data/models/tf_model_{self.seed}.ckpt")

    def restore(self, ckpt_loc: str):
        self.saver.restore(self.sess, ckpt_loc)
