
# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #raise valueerror if s is too big
    if s > la.matrix_rank(A):
        raise ValueError("s is greater than the number of nonzero singular values of A")

    U, sig, V_h = la.svd(A,full_matrices = False)
    U1 = U[:,0:s]
    m_U1, n_U1 = U1.shape

    sig1 = np.diag(sig[0:s])

    V1_h = V_h[0:s,:]
    m_V_h, n_V_h = V1_h.shape
    A_hat = U1 @ sig1 @ V1_h
    return A_hat, (m_U1*s+s+n_V_h*s)

def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, sig, V_h = la.svd(A,full_matrices = False)
    #if a minimum sigular value is less then an error raise ValueError
    if min(sig) > err:
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank.")
    s = np.argmax(sig<err)
    return svd_approx(A,s)

from imageio import imread
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank

def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    return:
        image_s (n array): image after compression
        probability (int): reduced/original
    """
    
    ax1 = plt.subplot(1,2,1)
    plt.axis("off")
    ax2 = plt.subplot(1,2,2)

    image = imread(filename)/255

    #when it is color:
    if(len(image.shape)==3):
        ax1.imshow(image)
        image_m,image_n, image_k = image.shape
        original_size = image_m*image_n*image_k
        Rs,n1 = svd_approx(image[:,:,0],s)
        Gs,n2 = svd_approx(image[:,:,1],s)
        Bs,n3 = svd_approx(image[:,:,2],s)

        #size for reduced
        n = n1+n2+n3

        Rs = np.clip(Rs,0,1)
        Gs = np.clip(Gs,0,1)
        Bs = np.clip(Bs,0,1)
        image_s = np.dstack((Rs,Gs,Bs))
        plt.axis("off")
        ax2.imshow(image_s)
    else:#if it is gray
        ax1.imshow(image,cmap='gray')
        image_m,image_n = image.shape
        original_size = image_m*image_n
        As,n = svd_approx(image,s)
        image_s = As
        ax2.imshow(image_s,cmap='gray')
    #calculate difference
    diff = original_size - n
    
    return image_s, n/original_size
    #raise NotImplementedError("Problem 5 Incomplete")

def animate_images(images,sizes, rank):
    """Animate a sequence of images. The input is a list where each
    entry is an array that will be one frame of the animation.
    """
    fig, ax = plt.subplots()

    plt.axis("off")
    im = plt.imshow(images[0], animated=True)
    title = ax.text(.5,.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    subtitle = ax.text(.5,.8,"",bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes,fontsize=10,ha='center')
    def data_gen(t=0):
        index = 0
        yield index
        while True:
            if index == len(sizes)-1:
                index=0
            else:
                index += 1
            yield index

    def update(data):
        index = data
        title.set_text("Rank {} Approximation".format(rank[index]))
        subtitle.set_text('size compared to the original:%.02f %%' %(sizes[index]*100))
        im.set_array(images[index])
        return im,title, subtitle, # Note the comma!
    ani = FuncAnimation(fig, update, data_gen, interval=250, repeat=True)
    ani.save("./output.gif", writer='imagemagick', fps=10, bitrate=-1)

    plt.show()




if __name__ == "__main__":
    A = np.array([[9,7,8],[3,4,1],[7,7,3]])
    print(compact_svd(A))

    #get images
    rank = [i*10 for i in range(1,16)]
    images = []
    sizes = []
    print(compress_image("hubble.jpg", 20))
    plt.show()
    """
    for i in rank:
        reduced_image, reduced_size = compress_image("hubble.jpg", i)
        images.append(reduced_image)
        sizes.append(reduced_size)
        print("rank: ",i," calculated")
    print(sizes)

    #we append last to give the dealy before repeat
    images.append(reduced_image)
    sizes.append(reduced_size)
    rank.append(rank[-1])
    images.append(reduced_image)
    sizes.append(reduced_size)
    rank.append(rank[-1])

    animate_images(images,sizes, rank)
    """
