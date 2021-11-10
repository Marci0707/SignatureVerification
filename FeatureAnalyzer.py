import numpy as np
import pandas as pd


class SignatureFeatureAnalyzer:
    """data:
        84 <--number of points
        2933 5678 31275775 0 1550 710 439  <-- data point [x,y,times stamp, pen down/up , azimuth, altitude, pressure]
        2933 5678 31275785 1 1480 770 420
        3001 5851 31275795 1 1350 830 433
        .
        .
        .
    """

    def __init__(self, data):
        features = ["x", "y", "time", "state", "azimuth", "altitude", "pressure"]
        self.df = pd.DataFrame(data, columns=features)
        self.df["y_d"] = (self.df["y"] - self.df["y"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))
        self.df["x_d"] = (self.df["x"] - self.df["x"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))
        self.df["y_d_d"] = (self.df["y_d"] - self.df["y_d"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))
        self.df["x_d_d"] = (self.df["x_d"] - self.df["x_d"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))
        self.df["y_d_d_d"] = (self.df["y_d_d"] - self.df["y_d_d"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))
        self.df["x_d_d_d"] = (self.df["x_d_d"] - self.df["x_d_d"].shift(1)) / (self.df["t"] - self.df["t"].shift(1))

    def calc_duration(self):
        return self.df["t"].iloc[-1] - self.df["t"].iloc[0]

    def calc_avg_jerk(self):
        self.df["abs_jerk"] = np.sqrt(self.df["x_d_d"] * self.df["x_d_d"] + self.df["y_d_d"] * self.df["y_d_d"])
        mean = self.df["abs_jerk"].mean()
        self.df.drop(["abs_jerk"], axis=1, inplace=True)
        return mean

    def calc_std_dev_in_col(self, col_name):
        return self.df[col_name].std()

    def count_local_maxima_in_col(self, col_name):
        c = 0
        col = self.df[col_name]
        for i in range(1, len(col) - 1):
            if col[i + 1] < col[i] > col[i - 1]:
                c += 1
        if col[0] > col[1]:
            c += 1
        if col[-1] > col[-2]:
            c += 2
        return c

    def f11(self):  # Jerk rms
        return (self.df["abs_jerk"] ** 2).mean() ** 0.5

    def f12(self):  #t(2nd pen_down) / duration
        pen_downs_found = 0
        for row_idx in range(len(self.df)):
            if self.df["state"].iloc[row_idx] == 0:
                pen_downs_found += 1
            if pen_downs_found == 2:
                return self.df["t"].iloc[row_idx] / self.calc_duration()

        return 1  # just in case

    def f13(self): #(avg abs(velocity)) / (maximum of Vx)
        return ((self.df["x_d"] ** 2 + self.df["y_d"] ** 2) ** 0.5).mean() / self.df["x_d"].max()

    def analyze(self):
        values = np.zeros(100)
        values[0] = self.calc_duration()
        values[1] = self.calc_pen_ups()
        values[2] = self.calc_sign_changes_in_col("x") + self.calc_sign_changes_in_col("y")
        values[3] = self.calc_avg_jerk()
        values[4] = self.calc_std_dev_in_col("y_d_d")
        values[5] = self.calc_std_dev_in_col("y_d")
        # values[6] = TODO
        values[7] = self.count_local_maxima_in_col("x")
        values[8] = self.calc_std_dev_in_col("x_d_d")
        values[9] = self.calc_std_dev_in_col("x_d")
        values[10] = self.f11()
        values[11] = self.count_local_maxima_in_col("y")
        values[12] = self.f12()
        values[13] = self.f13()
        #values[14] = TODO
        #values[15] = TODO
        #values[17] = TODO
        #values[18] = TODO
        #values[19] = TODO
        #values[20] = TODO

    def calc_sign_changes_in_col(self, col_name):
        col = self.df[col_name]
        c = 0
        for i in range(1, len(col) - 1):
            if col[i] < col[i + 1] != col[i - 1] < col[i]:
                c += 1
        return c

    def get_feature_list(self):
        return self.analyze()

    def calc_pen_ups(self):  # TODO nicer
        c = 0
        for point in self.df:
            if point[4] == 0:
                c += 1
        return c
