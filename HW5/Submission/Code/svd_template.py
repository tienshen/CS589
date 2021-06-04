import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def SVD(A, s, k):
    # Calculate probabilities p_i and use it as probibility distribution for constructing matrix S
    n,m = A.shape
    p = np.zeros(n)
    fro = np.linalg.norm(A[1:], ord = 'fro') ** 2
    for i in range(1, n):
        p[i] = (np.linalg.norm(A[i])**2)/fro
    # Construct S matrix of size s by m using p prob distribution
    S = np.zeros((s,m))
    for i in range(1,s):
        j = np.random.choice(n, p = p, replace=False)
        S[i] = A[j]
    # Calculate SS^T
    sst = np.matmul(S, S.T)

    # compute SVD for SS^T
    u, s , vh = np.linalg.svd(sst)
    lamb = np.sqrt(s[:k])

    # Construct H matrix of size m by k
    H = np.zeros((k,m))
    for i in range(1,k):
        st_w = np.matmul(S.T, vh[i])
        H[i] = st_w/np.linalg.norm(st_w)
    H = H.T
    print("shape of H: ", H.shape)

    # Return matrix H and top-k singular values sigma
    return H, lamb

def main():
    im = Image.open("../../Data/baboon.tiff")
    A = np.array(im)
    A = A.astype(np.float)
    n, m = A.shape
    s = 80
    k = 60
    H, sigma = SVD(A, s, k)


    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(im)
    # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.
    u, s, vh = np.linalg.svd(A)
    s_diag = np.diag(s)
    print("u: ", u.shape, "s: ", s_diag[:, :k].shape, "vh: ", vh[:k, :].shape)
    f_mat = np.matmul(u, s_diag[:, :k])
    A_optimal = np.matmul(f_mat, (vh[:k, :]))
    fig.add_subplot(1,3,2)
    plt.title("Optimal")
    plt.imshow(A_optimal)


    # TO DO: Use H to compute sub-optimal k rank approximation for A
    sigma = np.diag(sigma)
    print("H: ", H.shape, "sigma: ", sigma.shape, "HT: ", H.T.shape)
    A_sub_optimal = np.matmul(np.matmul(A, H), H.T)
    fig.add_subplot(1, 3, 3)
    plt.title("Sub-optimal")
    plt.imshow(A_sub_optimal)

    # To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation


    # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    # sub-optimal k-rank approximation for A using H.
    error_optimal = np.linalg.norm(A-A_optimal, ord = "fro")
    error_sub_optimal = np.linalg.norm(A-A_sub_optimal, ord = "fro")

    print("error for optimal: ", error_optimal, "\nerror for sub-optimal is: ", error_sub_optimal)



    plt.show()

if __name__ == "__main__":
    main()
