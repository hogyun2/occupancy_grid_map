from dis import dis
import scipy.io
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Map():

    def __init__(self, x_size, y_size, grid_size):

        self.x_size = x_size
        self.y_size = y_size
        
        self.grid_size = grid_size
                                        
        self.grid_position = np.array([np.tile(np.arange(0, self.x_size * self.grid_size, self.grid_size)[: ,None], (1, self.y_size)),              # 1 * 100 shape으로 계속 tile
                                       np.tile(np.arange(0, self.y_size * self.grid_size, self.grid_size)[: ,None].T, (self.x_size, 1) )])          # 100 * 1 shape으로 계속 tile

        # scale of obstacle
        self.alpha = 3.0
        
        # max theta(=threshold)
        self.beta = 5.0 * np.pi / 180.0
    
        # It is assumed that the probability of each cell is 0.65
        # Log odds notation
        # logit 변환
        self.log_free = np.log(0.35 / 0.65)
        self.log_occupy = np.log(0.65 / 0.35)
        
        
        # set all to zero
        self.log_prob_map = np.zeros((self.x_size, self.y_size))

    
    def update_map(self, pose, measurement):
             
        dx = self.grid_position.copy()

        dx[0, :, :] -= pose[0]

        # shape of dx[0, :, :]
        # [ 0.  0.  0. ...  0.  0.  0.]
        # [ 1.  1.  1. ...  1.  1.  1.]
        # [ 2.  2.  2. ...  2.  2.  2.]
        # ...
        # [97. 97. 97. ... 97. 97. 97.]
        # [98. 98. 98. ... 98. 98. 98.]
        # [99. 99. 99. ... 99. 99. 99.]
        
        dx[1, :, :] -= pose[1]

        # shape of dx[1, :, :]
        #   [ 0.  1.  2. ... 97. 98. 99.]
        #   [ 0.  1.  2. ... 97. 98. 99.]
        #   [ 0.  1.  2. ... 97. 98. 99.]
        #   ...
        #   [ 0.  1.  2. ... 97. 98. 99.]
        #   [ 0.  1.  2. ... 97. 98. 99.]
        #   [ 0.  1.  2. ... 97. 98. 99.]

        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2]

        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis = 0)
        
        for measurement_i in measurement:

            r = measurement_i[0]
            theta = measurement_i[1]

            free  = (np.abs(theta_to_grid - theta) <= self.beta / 2.0) & (dist_to_grid < (r - self.alpha / 2.0))
            occupy = (np.abs(theta_to_grid - theta) <= self.beta / 2.0) & (np.abs(dist_to_grid - r) <= self.alpha / 2.0)
            
            self.log_prob_map[free] += self.log_free
            self.log_prob_map[occupy] += self.log_occupy


if __name__ == '__main__':

    data = scipy.io.loadmat('./state_meas_data.mat')

    # state = [x, y, theta]
    state = data['X']

    # measurements
    # 제공하는 file의 measurements data는 FOV가 180
    # FOV를 11구간으로 나누고 각 구간에 들어오는 r, theta data가 input으로 들어옴
    measurement = data['z']
    
    grid_size = 1.0

    map = Map(int(100/grid_size), int(100/grid_size), grid_size)

    # real time plot
    plt.ion()
    plt.figure(1)

    for i in tqdm(range(len(state.T))):

        map.update_map(state[:,i], measurement[:,:,i].T)

        print(measurement[1, :, :])

        # Real-Time Plotting
        # comment out these next lines to make it run super fast, matplotlib is painfully slow
        plt.clf()

        pose = state[:,i]

        # add the polygon(shape of robot) in graph
        robot = plt.Circle((pose[1], pose[0]), radius = 3.0, fc = 'y')
        plt.gca().add_patch(robot)


        # add the arrow which demonstrated a direction of robot in
        # 2D rotation matrix에 scale을 곱해줌
        arrow = pose[0:2] + np.array([3.5, 0]).dot(np.array([[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]))
        plt.plot([pose[1], arrow[1]], [pose[0], arrow[0]])
        
        
        # 각 cell의 probability map = 1.0 - 1./(1.+np.exp(map.log_prob_map))
        plt.imshow(1.0 - 1/(1 + np.exp(map.log_prob_map)), 'Greys')
        
        # matplot libarary가 대기할 시간을 줌
        plt.pause(0.005)


    plt.ioff()
    plt.clf()
    plt.imshow(1.0 - 1/(1 + np.exp(map.log_prob_map)), 'Greys')
    # plt.imshow(map.log_prob_map, 'Greys')
    plt.show()