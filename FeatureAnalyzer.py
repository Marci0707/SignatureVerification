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

    def f13(self):  # t(2nd pen_down) / duration
        pen_downs_found = 0
        for row_idx in range(len(self.df)):
            if self.df["state"].iloc[row_idx] == 0:
                pen_downs_found += 1
            if pen_downs_found == 2:
                return self.df["t"].iloc[row_idx] / self.calc_duration()

        return 1  # just in case

    def f14(self):  # (avg abs(velocity)) / (maximum of Vx)
        return ((self.df["x_d"] ** 2 + self.df["y_d"] ** 2) ** 0.5).mean() / self.df["x_d"].max()

    def delta(self, col_idx):
        sum_ = 0
        state_idx = 3
        curr_max = self.df.loc[1][col_idx]
        curr_min = self.df.loc[1][col_idx]
        for record in self.df.iterrows():
            if record[1][state_idx] == 0:
                sum_ += curr_max - curr_min
                curr_max = -np.inf
                curr_min = np.inf
            if record[1][col_idx] > curr_max:
                curr_max = record[1][col_idx]
            if record[1][col_idx] < curr_min:
                curr_min = record[1][col_idx]

    def f15(self):
        A_min = (self.df['y'].max() - self.df['y'].min()) * (self.df['x'].max() - self.df['x'].min())
        denominator = self.delta(0) * self.delta(1)
        return A_min / denominator

    def f7(self):
        return self.calc_std_dev_in_col('y') / self.delta(1)

    def f16_and_18(self):
        for i in range(len(self.df) - 1, -1, -1):
            if self.df['state'].iloc[i] == 0:
                x_last_penup = self.df['x'].iloc[i]
                y_last_penup = self.df['y'].iloc[i]
                x_max = self.df['x'].max()
                y_min = self.df['y'].min()
                f16 = (x_last_penup - x_max) / self.delta(0)
                f18 = (y_last_penup - y_min) / self.delta(1)
                return f16, f18

    def f17_and_f19(self):
        for i in range(len(self.df)):
            if self.df['state'].iloc[i] == 1:
                x_first_pendown = self.df['x'].iloc[i]
                y_first_pendown = self.df['y'].iloc[i]
                x_max = self.df['x'].max()
                y_min = self.df['y'].min()
                f17 = (x_first_pendown - x_max) / self.delta(0)
                f19 = (y_first_pendown - y_min) / self.delta(1)
                return f17, f19

    def f22(self):
        indices = self.df.index[self.df['state'] == 0].tolist()
        pendown_duration = 0
        for i in indices:
            pendown_duration += self.df['t'].loc[i + 1] - self.df['t'].loc[i]

        return pendown_duration / self.calc_duration()

    def analyze(self):
        values = np.zeros(20)
        values[0] = self.calc_duration()
        values[1] = self.calc_pen_ups()
        values[2] = self.calc_sign_changes_in_col("x") + self.calc_sign_changes_in_col("y")
        values[3] = self.calc_avg_jerk()
        values[4] = self.calc_std_dev_in_col("y_d_d")
        values[5] = self.calc_std_dev_in_col("y_d")
        values[6] = self.f7()
        values[7] = self.count_local_maxima_in_col("x")
        values[8] = self.calc_std_dev_in_col("x_d_d")
        values[9] = self.calc_std_dev_in_col("x_d")
        values[10] = self.f11()
        values[11] = self.count_local_maxima_in_col("y")
        values[12] = self.f13()
        values[13] = self.f14()
        values[14] = self.f15()
        values[15], values[17] = self.f16_and_18()
        values[16], values[18] = self.f17_and_f19()
        values[19] = self.f22()

        return values

    def calc_sign_changes_in_col(self, col_name):
        col = self.df[col_name]
        c = 0
        for i in range(1, len(col) - 1):
            if col[i] < col[i + 1] != col[i - 1] < col[i]:
                c += 1
        return c

    def get_feature_list(self):
        return self.analyze()

    def calc_pen_ups(self):
        return len(self.df.index[self.df['state'] == 0].tolist())
