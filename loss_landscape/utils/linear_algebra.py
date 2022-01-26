import numpy
import torch
from tqdm import tqdm


class FrequentDirectionAccountant:
    """
    Frequent Directions algorithm (Alg 2 from the paper) for streaming SVD.
    """

    def __init__(self, k, l, n, device):
        """

        :param k: number of eigen vectors we want eventually (k should be less than l+1)
        :param l: buffer size
        :param n: number of parameters/dimension of vector
        :param device:
        """
        self.K = k
        self.L = l
        self.N = n

        self.step = 0
        self.buffer = torch.zeros(self.L, self.N, device=device)

    def update(self, vector):
        """
        run one step of Freq Direction
        :param vector:
        :return:
        """

        self.buffer[self.L - 1] = vector
        _, S, Vt = torch.linalg.svd(self.buffer, full_matrices=False)
        delta = S[-1] ** 2
        new_svd_vals = torch.sqrt(torch.clip(S ** 2 - delta, min=0, max=None))
        self.buffer = torch.diag(new_svd_vals) @ Vt
        self.step += 1

    def get_current_buffer(self):
        return self.buffer

    def get_current_directions(self):
        """return top k eigen vectors of A^TA"""
        _, _, Vt_B = torch.linalg.svd(self.buffer, full_matrices=False)
        return Vt_B[:self.K]


if __name__ == "__main__":

    def test(A):
        K = 2
        L = 4
        m, n = A.shape

        fd = FrequentDirectionAccountant(k=K, l=L, n=n, device="cpu")

        for i in tqdm(range(m)):
            fd.update(A[i])

        B = fd.get_current_buffer()

        # Verify the theorems
        AtA = numpy.transpose(A) @ A
        BtB = numpy.transpose(B) @ B

        # Theorem 1.1
        print("Checking Theorem 1.1")
        diff = AtA - BtB
        _, S_diff, _ = numpy.linalg.svd(diff)

        U, S, Vt = numpy.linalg.svd(A, full_matrices=False)
        Ak = U[:, :K] @ numpy.diag(S[:K]) @ Vt[:K, :]

        frob_norm = numpy.linalg.norm(A - Ak)

        if S_diff.max() < frob_norm ** 2 / (L - K):
            print("Theorem 1.1 passed")
        else:
            print("Theorem 1.1 failed")

        print("Checking Theorem 1.2")
        _, _, Vt_B = numpy.linalg.svd(B)

        pi_B_A = A @ Vt_B @ numpy.transpose(Vt_B)
        proj_frob_norm = numpy.linalg.norm(A - pi_B_A)

        if proj_frob_norm ** 2 < L * frob_norm ** 2 / (L - K):
            print("Theorem 1.2 passed")
        else:
            print("Theorem 1.2 failed")


    # we create a huge matrix with large number of rows
    A = numpy.random.rand(20000, 100)
    test(A)

    # In actual problem, we are going to have large rows and small column
    A = numpy.random.rand(20, 10000)
    test(A)
