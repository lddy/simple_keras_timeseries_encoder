import pandas as pd
import numpy as np
import random
from collections import deque

class neighbour:
    def __init__(self, data):
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.data['picnic'] = 0
        self.data['new_hat'] = 0
        self.data['hat'] = 0
        self.data['WeekendTomorrow'] = 0
        self.data['naive_prediction'] = 0
        self.data_diag = data.copy()
        self.data['hat'] = 0
        self.data['new_hat'] = 0
        self.data_diag['hat'] = 0
        self.data_diag['conditions'] = 'good'
        self.data_diag['is_picnic'] = False
        self.data_diag['is_work'] = False
        self.data_diag['prob_cals'] = 0.0
        self.data_diag['picnic_roll'] = False
        self.data_diag['rand_roll'] = 0.0
        self.data_diag['medium_roll'] = False
        self.data_diag['block_picnic_due_to_last'] = False
        self.rand_factor = 0.01
        self.picnic_chance = {
            4: 0.4,
            5: 0.6,
            6: 0.75,
            7: 0.75,
            8: 0.75,
            9: 0.5,
            10: 0.2
        }
        self.temp_cold = 55
        self.temp_cold_chance = 0.5
        self.temp_very_cold = 38
        self.rain = 0.8
        self.light_rain = 0.25
        self.light_rain_chance = 0.3
        self.curr_day = 0
        self.last_weekend_picnic = -1
        self.last_picnic_date = -1
        self.newstate = True
        self.newevent = 0
        self.newreset = 14
        self.newchance = 0.05
        self.histlength = 3
        self.hat_hist = 0
        self.max_point = len(data) - 1

    def run_all(self):

        for i in range(self.max_point):
            #populate naive prediction
            is_naive_picnic = False
            if i < self.max_point - 1:
                if self.data.at[i+1, 'MOY'] in self.picnic_chance and\
                        self.picnic_chance[self.data.at[i+1, 'MOY']] >= 0.5\
                        and  self.data.at[i+1, 'DOW'] in [7,1]:
                    is_naive_picnic = True
            is_naive_work = False
            if 1 < self.data.at[i+1, 'DOW'] < 7:
                is_naive_work = True
            is_naive_bad_weather = False
            if (self.data.at[i+1, 'Temp_NextForecast'] <= self.temp_very_cold or self.data.at[i, 'Prec_NextForecast'] > self.rain):# or\
               # (self.data.at[i+1, 'Temp_NextForecast'] <= self.temp_cold or self.data.at[i, 'Prec_NextForecast'] > self.light_rain):
                is_naive_bad_weather = 'true'

            if (is_naive_work or is_naive_picnic) and is_naive_bad_weather:
                self.data.at[i+1, 'naive_prediction'] = 1
                self.data_diag.at[i+1, 'naive_prediction'] = 1
            else:
                self.data.at[i+1, 'naive_prediction'] = 0
                self.data_diag.at[i+1, 'naive_prediction'] = 0


            #maintain states
            if self.newstate and i - self.newevent > self.newreset:
                self.newstate = False
            if not self.newstate and random.uniform(0.0, 1.0) <= self.newchance + (i - self.newevent)/(self.newreset*2.0):
                self.newstate = True
                self.newevent = i
                self.data.at[i, 'new_hat'] = 1
                self.data_diag.at[i, 'new_hat'] = 1

            block_picnic_due_to_last = False
            if (self.data.at[i, 'DOW'] in [6, 7]):
                self.data.at[i, 'WeekendTomorrow'] = 1
                self.data_diag.at[i, 'WeekendTomorrow'] = 1
            if (self.data.at[i, 'DOW'] == 7 or (i > 0 and self.data.at[i, 'HolidayTomorrow'])) and\
                    i-self.last_picnic_date < 9 or\
                self.data.at[i, 'DOW'] == 1 and 1 < i-self.last_picnic_date < 9:
                block_picnic_due_to_last = True
            self.data_diag.at[i, 'block_picnic_due_to_last'] = block_picnic_due_to_last

            conditions = 'good'
            prob_cals = 0.0
            #compute current conditions
            if self.data.at[i, 'Temp'] <= self.temp_very_cold or self.data.at[i, 'Precipitation'] > self.rain:
                conditions = 'bad'
            elif self.data.at[i, 'Temp'] <= self.temp_cold or self.data.at[i, 'Precipitation'] > self.light_rain:
                conditions = 'medium'
                if self.data.at[i, 'Precipitation'] > self.light_rain:
                    prob_cals += self.light_rain_chance
                if  self.data.at[i, 'Temp'] <= self.temp_cold:
                    prob_cals += self.temp_cold_chance
            self.data_diag.at[i, 'prob_cals'] = prob_cals
            rand_dice_roll = random.uniform(0, 1) <= self.rand_factor
            self.data_diag['rand_roll'] = rand_dice_roll
            medium_roll = random.uniform(0, 1) < prob_cals
            self.data_diag.at[i, 'medium_roll'] = medium_roll

            #compute picnic state
            is_picnic = False
            picnic_roll = False
            is_work = False
            if 1 < self.data.at[i, 'DOW'] < 7 and  (i == 0 or self.data.at[i - 1, 'HolidayTomorrow'] == 0):
                is_work = True
            if not is_work and self.data.at[i, 'MOY'] in self.picnic_chance and not block_picnic_due_to_last:
                picnic_roll =random.uniform(0.0, 1.0) < self.picnic_chance[self.data.at[i, 'MOY']]
                if picnic_roll:
                    is_picnic = True
                self.data_diag.at[i, 'picnic_roll'] = picnic_roll

            self.data_diag.at[i, 'is_picnic'] = is_picnic
            self.data_diag.at[i, 'is_work'] = is_work
            self.data.at[i, 'picnic'] = 1.0 if is_picnic else 0.0

            hat = False
            #process work day
            if is_work:
                if conditions == 'bad':
                    hat = True
                elif conditions == 'medium':
                    if self.newstate:
                        hat = True
                    elif self.hat_hist < self.histlength:
                        hat = medium_roll
                    elif self.hat_hist >= self.histlength:
                        hat = False
                        self.hat_hist = 0
                    else:
                        hat = False
                else:
                    if self.hat_hist >= self.histlength:
                        self.hat_hist = 0
                    if rand_dice_roll and self.hat_hist < self.histlength:
                        hat = False
            elif is_picnic:
                self.last_picnic_date = i
                if conditions == 'bad':
                    hat = True
                elif conditions == 'medium':
                    if picnic_roll:
                        if self.newstate:
                            hat = True
                        elif self.hat_hist < self.histlength:
                            hat = medium_roll
                        elif self.hat_hist >= self.histlength:
                            hat = False
                            self.hat_hist = 0
                        else:
                            hat = False
            else:#weekend
                if self.hat_hist < self.histlength:
                    hat = rand_dice_roll

            if hat == True:
                self.data.at[i, 'hat'] = 1
                self.data_diag.at[i, 'hat'] = 1
                self.hat_hist += 1








