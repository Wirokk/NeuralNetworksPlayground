import numpy as np
import mnist_loader

class Network(object): 

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z): 
        return 1/(1+np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a) + b # Similar to perceptron output
            a = self.sigmoid(a) # Output compression between 0-1
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, log_fn=print, epoch_callback=None):
        """
        Stochastic Gradient Descent.

        - training_data : liste [(x, y), ...]
        - epochs        : nb d'epochs
        - mini_batch_size
        - eta           : learning rate
        - test_data     : pour évaluer après chaque epoch
        - log_fn(msg)   : fonction pour logger (par défaut print)
        - epoch_callback(epoch, metrics, network) : callback appelé en fin d'epoch
        """

        if log_fn is None:
            def log_fn(msg):
                pass

        log_fn("Starting SGD training...")
        if test_data:
            n_test = len(test_data)
        else:
            n_test = None

        n = len(training_data)

        for j in range(epochs):
            # shuffle
            np.random.shuffle(training_data)

            # mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # update for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # metrics de fin d'epoch
            metrics = {"epoch": j + 1, "epochs": epochs}

            if test_data:
                correct = self.evaluate(test_data)
                acc = correct / n_test
                metrics.update({
                    "test_correct": correct,
                    "test_total": n_test,
                    "test_accuracy": acc,
                })
                log_fn(
                    "Epoch {0}/{1}: {2} / {3} correct (accuracy={4:.4f})".format(
                        j + 1, epochs, correct, n_test, acc
                    )
                )
            else:
                log_fn("Epoch {0}/{1} complete".format(j + 1, epochs))

            # callback pour l'UI (graph, poids, etc.)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch=j + 1, metrics=metrics, network=self)
                except Exception as e:
                    log_fn(f"[epoch_callback error] {e}")

    def update_mini_batch(self, mini_batch, eta): # Update the networks's weights and biases by applying gradient descent
        nabla_b = [np.zeros(b.shape) for b in self.biases] # Create zero arrays w/ same shape as the biases vector. 
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Create zero arrays w/ same shape as the weights vector
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # Add the newly calculated gradients 
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    
    def backpropagation(self, x, y): # backpropagation for the neural networks

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        #Return the vector of partial derivatives \partial C_x partial a for the output activations.
        return (output_activations-y)
    
def main():
    net = Network([784, 100, 10])  # 784 input neurons, 100 hidden neurons, 10 output neurons
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
    pass


if __name__ == '__main__':
    main()