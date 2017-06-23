import caffe

caffe.set_mode_cpu()

solver_file = "nets/my_solver.prototxt"
solver = caffe.SGDSolver(solver_file) # stochastic gradient descent

solver.solve()
# note: the net architecture is specified in the solver.prototxt
# and the training and testing sets are specified in the
# netDescription.prototxt!
