class UGME(object):
    def __init__(self, tsteps=1, rd=1, dt=1.0):
        self.dt = dt
        self.tsteps = tsteps
        self.rd = rd
        self.exact = np.zeros((tsteps, rd, rd))
        self.lag_time = np.zeros(tsteps)
        self.tau_r = 0
        self.tau_R = 0
        self.cutoff_list = np.zeros(tsteps)
        self.U_matrix = np.zeros(tsteps)
        self.rmses = np.zeros(tsteps)
        self.__row_norm = False
        self.__col_norm = False
        self.__get_data = False
        self.__get_dTPM_dt = False


    def GetData(self, input_data):
        # load data and find out which normalization
        temp = np.load(input_data, allow_pickle=True) #[0] may need to remove this
        temp = np.einsum("ijk->kji", temp)
        #print(temp.shape)
        self.rd, self.tsteps = temp[0].shape

        # add identity matrix
        self.tsteps += 1
        self.exact = np.zeros([self.rd, self.rd, self.tsteps])
        self.exact[:, :, 0] = np.eye(self.rd)
        self.exact[:, :, 1:] = np.copy(temp)
        
        return self.exact

    def which_normalization(self):
        """
        Check if a time-dependent matrix is row or column normalized across time steps.

        Parameters:
        -----------
        TPM : numpy.ndarray
            A time-dependent matrix with shape [time_steps, rows, columns].

        Returns:
        --------
        int
            Returns 0 if the matrix is column-normalized across time steps,
            Returns 1 if the matrix is not column-normalized across time steps.
            Prints the normalization status for rows and columns across time steps.

        This function calculates row and column sums across the specified axes for each time step 
        and checks if the sums are approximately equal to 1, indicating row or column normalization.
        """
        # Normalize along rows and columns
        row_sums = np.sum(self.exact, axis=1)  # Sum across columns for each time step
        col_sums = np.sum(self.exact, axis=0)  # Sum across rows for each time step

        # Check if rows sum up to 1 for each time step (row normalization)
        row_normalized = np.all(np.isclose(row_sums, 1.0))

        # Check if columns sum up to 1 for each time step (column normalization)
        col_normalized = np.all(np.isclose(col_sums, 1.0))

        if row_normalized:
            print("The matrix is row-normalized at all times.")
            

        if col_normalized:
            print("The matrix is column-normalized at all times.")
            return 0

    def Get_U(self):
        self.U_matrix = np.zeros_like(self.exact)

        # initial value
        self.U_matrix[:, :, 0] = np.copy(self.exact[:, :, 0])

        # get the inverse
        for k in range(1, self.tsteps):
                    self.U_matrix[:, :, k] = np.dot(self.exact[:, :, k],
                                                    np.linalg.inv(self.exact[:, :, k-1]))
    
        return self.U_matrix

    def GetAve(self, t_r, t_R):
        return np.average(self.U_matrix[:, :, t_r:t_R], axis=-1)

    def GetCfromU(self, U_inf, tau_R, long_tsteps=1):
        upper_limit = int(np.max((self.tsteps, long_tsteps)))
        Ct = np.zeros_like(self.exact)
        Ct[:, :, 0] = np.eye(self.rd)
        for k in range(1, tau_R): Ct[:, :, k] = np.copy(self.exact[:, :, k])
        for k in range(tau_R, upper_limit): Ct[:, :, k] = U_inf @ Ct[:, :, k-1]
        return Ct

    def rmse(self, matrix):
        total = 0.0
        for k in range(self.tsteps):
            difference  = self.exact[:, :, k] - matrix[:, :, k]
            difference_squared = difference**2
            total += np.sum(difference_squared)
        total = (total / (self.tsteps * self.rd**2))**0.5
        return total

    def rmse_only_tau_R(self, cutoff_list, save=1):
        """
        This does not use the onset of averaging parameter
        only uses the final cutoff times tR
        """
        if save:
            self.rmses = []
            self.cutoff_list = cutoff_list
        for tR in cutoff_list:
            tR = int(tR)
            U_inf = self.U_matrix[:, :, int(tR)]
            prediction = self.GetCfromU(U_inf, tR)
            self.rmses.append(self.rmse(prediction))
        return self.rmses
        

    def rmse_with_tr(self, tr_list):
        """
        This DOES use the onset of averaging parameter, tr
        """
        tr_rmses_list = []

        for tr in tr_list:
            tr = int(tr)
            new_cutoff_list = self.cutoff_list[self.cutoff_list > tr]

            tR_rmses = []
            for tR in new_cutoff_list:
                tR = int(tR)
                U_inf = self.GetAve(tr, tR)
                prediction = self.GetCfromU(U_inf, tR)
                tR_rmses.append(self.rmse(prediction))
            tr_rmses_list.append(np.array(tR_rmses))

        return tr_rmses_list