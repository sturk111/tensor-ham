import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.losses import Loss
from tensorflow.keras import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.ion()

k_vec = pickle.load(open('/Users/simonturkel/Documents/Python/code_folio/tensorbands/k_vec.pkl','rb')) #array of momentum vectors along which the bands are computed
evals = pickle.load(open('/Users/simonturkel/Documents/Python/code_folio/tensorbands/evals.pkl','rb')) #eigenvalues associated with each point in momentum space

#orbital positions in reduced coordinates
orbs = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[2/3,1/3],[2/3,1/3],[2/3,1/3],[1/3,2/3],[1/3,2/3],[1/3,2/3]])

#define an array of displacement vectors between each pair of orbitals
Rij = np.zeros((len(orbs),len(orbs),2))
for i in range(len(orbs)):
	for j in range(len(orbs)):
		Rij[i,j,:] = -orbs[i,:] + orbs[j,:]

#an array of neighboring unit cells
cells = np.array([[0,0],[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]])

class GenHam(Layer):
	def __init__(self, Rij, cells):
		'''
		This custom layer creates a Hamiltonian with orbital structure given by the two input arrays Rij and cells.
		The hoppings are initialized randomly and are set to be trainable.
		The call function returns the eigenvalues of the Hamiltonian in sorted order.

		Parameters
		----------
		Rij : numpy array
			This array stores the vector displacements between all pairs of orbitals.  
			The shape is (numOrbs, numOrbs, dim), where numOrbs is the number of orbitals 
			and dim is the number of spatial dimensions (dim = 2 in our example).
		cells : numpy array
			This array contains a list of cells to include in the hoppings.  In our example we include 
			the home unit cell (R=0) and the six nearest neighbor cells on a hexagonal lattice.
		'''
		super(GenHam, self).__init__()
		#this defines the geometry of the lattice
		self.Rij = Rij
		self.cells = cells
	
	def build(self,input_shape):
		#Build the layer by randomly initializing the hopping amplitudes in self.amp and storing Rij as a tensor
		w_init = tf.random_normal_initializer()
		#self.amp is the array of hoppings that we would like to learn for each orbital i,j and for each cell.
		self.amp = tf.Variable(name='amplitude_matrix', initial_value = w_init(shape = (self.Rij.shape[0],self.Rij.shape[1],int((self.cells.shape[0]+1)/2))), dtype = 'float32', trainable=True)
		#self.Rijvar is a tensor containing information about the lattice for later use
		#if we set trainable to True, then it will attempt to learn the lattice geometry as well
		self.Rijvar = tf.Variable(name='orbitals', initial_value = self.Rij, dtype = 'float32', trainable=False)
	
	def call(self, k):
		'''
		Construct the Hamiltonian at the given k point and solve for the resulting eignevalue spectrum.

		Parameters
		----------
		k : numpy array
			The momentum point at which to construct the Hamiltonian in reduced coordinates.
			The shape is (dimk,), where dim is the number of momentum space dimensions (dimk = 2 in our example).

		Returns
		-------
		evals : tf.Tensor shape = (numOrbs, ), dtype = complex64
			A tensor containing the (possibly complex) eigenvalues sorted according to the real part. 
		'''
		k = tf.transpose(k)
		R = tf.stack([self.Rijvar for _ in range(self.cells.shape[0])],axis=2) + self.cells
		
		#hoppings must be symmetric
		A0 = self.amp[:,:,0] + tf.transpose(self.amp[:,:,0])
		A1 = self.amp[:,:,1]
		A2 = self.amp[:,:,2]
		A3 = self.amp[:,:,3]
		A4 = tf.transpose(A1)
		A5 = tf.transpose(A2)
		A6 = tf.transpose(A3)
		A = tf.stack([A0,A1,A2,A3,A4,A5,A6],axis=2)

		tmp = tf.cast(tf.squeeze(tf.matmul(R,k)),'complex64')
		phase_mat = tf.exp(2*1j*np.pi*tmp)
		ham = tf.reduce_sum(tf.multiply(tf.cast(A,'complex64'),phase_mat),axis=-1)
		e,v  = tf.linalg.eig(ham)
		
		evals = tf.experimental.numpy.take(e,tf.argsort(tf.math.real(e)))
		return evals

class LeastSquares(Loss):
	def __init__(self):
		'''
		A custom loss function to quantify the difference between the learned eigenvalue spectrum 
		and the precomputed eigenvalue spectrum.
		'''
		super().__init__()

	#expect y_true and y_pred to be (numOrbs,) eigenvalues for a given k point
	def call(self, y_true, y_pred):
		'''
		Calculates the sum of squared differences between the predicted and true eigenvalue spectrum,
		masked to a window of +/- 1.5 eV around the zero of energy.  An additional term proportional
		to the imaginary part of the spectrum is added to push the learned hoppings towards hermiticity.

		Parameters
		----------
		y_true : tf.Tensor shape = (numOrbs, ), dtype = complex64
			True eigenvalues from precomputed band structure.
		y_pred : tf.Tensor shape = (numOrbs, ), dtype = complex64
			Predicted eigenvalues from learned model

		Returns
		-------
		loss : float32
			The computed loss for given y_treu and y_pred.
		'''
		mask = tf.cast(tf.abs(y_true)<=1.5,'float32')
		e_real = tf.math.real(y_pred)
		e_imag = tf.math.imag(y_pred)

		squared_error = tf.reduce_sum(tf.multiply(tf.math.pow(tf.abs(e_real - y_true),2),mask))
		
		loss = 1.5*squared_error + tf.reduce_sum(tf.square(e_imag))
		
		return loss

#Define and compile the model with our custom GenHam layer and LeastSquares loss
model = tf.keras.Sequential([GenHam(Rij, cells)]) 
opt = tf.keras.optimizers.SGD(learning_rate=5e-3)
model.compile(optimizer=opt, loss=LeastSquares())

#Fit the model to the precomputed band structure over 50 epochs
model.fit(k_vec,evals.T,epochs=50,batch_size=1)

#Calculate the band structure using the learned model and save the result to the array evals_model
evals_model = np.zeros(evals.shape)
for i in range(len(k_vec)):
	e = tf.math.real(model(np.array([[k_vec[i,0],k_vec[i,1]]]))).numpy()
	evals_model[:,i] = e

#Plot a comparison between the learned model and the true band structure
for i in range(evals.shape[0]):
	plt.plot(evals[i,:],color='k')
	plt.plot(evals_model[i,:],color='r',linestyle='--')
plt.ylim([-1,1])
plt.xlim([0,128])
