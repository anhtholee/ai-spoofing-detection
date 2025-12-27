## Metrics

### Precision, recall and F1 score

    ML - Precision: 0.9378, Recall: 0.6526, F1 Score: 0.7696
    Rules - Precision: 0.4602, Recall: 0.7507, F1 Score: 0.5706
    Hybrid - Precision: 0.4602, Recall: 0.7507, F1 Score: 0.5706


Some observation
- ML model has the best precision (94%), which means the flagged data are very likely to be spoofed. But the recall is only 65%, meaning quite a large number of spoofed events are not detected.
- Rules based model on the other hand has better recall (75%), meaning it successfully caught a larger majority of spoofing events, at least compared to ML model. However it has more false positives (low precision of 46%), which means a lot of false alarms, which can affect legitimate users.

Why do rules-based method and hybrid method have the exact same metrics? 


Let's see the mean of the scores by ML model




    np.float64(0.2853066666666667)



And the mean scores of rules-based model




    np.float64(0.565)



- The rules-based system outputs a score of 1.0 whenever any rule is triggered. Since rules based approach have a high recall (0.75) but low precision, it means the rules are triggered frequently, which results in many 1s. 
- The hybrid score is a simple average of ML score and rules score. Whenever the rule is triggered (rules score = 1), the hybrid score will be `(ml_score + 1) / 2`. Since ML score is always between 0 and 1, the hybrid score will always be greater than 0.5. Then with thresholding level of 0.5, it will be flagged as spoofed. So basically the hybrid model is defaulting to the output of the rules-based model.

For a better hybrid model, we can
- Tune the weighted average instead of using 50/50. This would make the ML score more influential and can likely improve precision.
- Use rules-based model as first pass and ML as second pass. If rules-based model triggered, we will use ML score, otherwise just use rules-based.
- Use triggered rules as a feature for the ML model.


### PR Curve


    
![png](eval_report_files/eval_report_18_0.png)
    


From the PR curve, we can see
- ML model outperforms the rules-based model significantly (0.96 vs 0.65 in AUC). The curve for ML model is consistently higher and to the right, which means better precision at every threshold.
- There is a large gap between the 2 curves, which is expected. ML model might be capturing complex sensor noise or patterns that simple rules cannot
- The ML curve maintains high precision at lower recall levels, which means that the ML model is good at identifying obvious spoofing events. 
- However precision drops quite sharply after recall = 0.7. This suggests there are some complex spoofing events that are harder to distinguish.

## Error analysis

False positives examples




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_id</th>
      <th>spoof_score_ml</th>
      <th>spoofed</th>
      <th>is_spoofed_ground_truth</th>
      <th>true_label</th>
      <th>installation_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>timestamp_unix</th>
      <th>horizontal_accuracy</th>
      <th>altitude</th>
      <th>speed</th>
      <th>bearing</th>
      <th>pressure_hpa</th>
      <th>mock_location_enabled</th>
      <th>device_is_charging</th>
      <th>wifi_bssid</th>
      <th>cell_tower_id</th>
      <th>num_satellites</th>
      <th>vertical_accuracy</th>
      <th>ambient_light_lux</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d436e615-e640-4b28-8f9a-34e51b5c92e5</td>
      <td>0.40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>875ddf9e-fcde-453b-8349-edf148fb6951</td>
      <td>40.796700</td>
      <td>-74.013831</td>
      <td>1766743638</td>
      <td>17.031549</td>
      <td>39.078199</td>
      <td>1.466237</td>
      <td>147.667885</td>
      <td>1009.672435</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-9081-7859</td>
      <td>20</td>
      <td>20.778224</td>
      <td>220.336131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bc22ccc1-8da8-41d0-83ea-52c8e446d461</td>
      <td>0.38</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>71e6def5-e53b-4441-84b3-fe70494e5567</td>
      <td>40.616599</td>
      <td>-74.007260</td>
      <td>1766743448</td>
      <td>2.000000</td>
      <td>53.719919</td>
      <td>1.176676</td>
      <td>131.986992</td>
      <td>1006.092087</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-4252-6804</td>
      <td>14</td>
      <td>23.992997</td>
      <td>131.026539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fe02a8a2-e82d-419e-ae1e-ec428f446191</td>
      <td>0.36</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>b589f403-b6e6-46f9-a215-c073ad7069fb</td>
      <td>40.703714</td>
      <td>-73.963429</td>
      <td>1766743578</td>
      <td>13.633510</td>
      <td>50.761166</td>
      <td>1.385804</td>
      <td>280.240889</td>
      <td>1006.867714</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-4083-4722</td>
      <td>12</td>
      <td>10.983824</td>
      <td>249.822988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9f66bc2e-f2b7-4f4d-b02c-88149bf414f7</td>
      <td>0.46</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>e94d5b0f-2dc3-48ab-b3f5-ce7ba45862d0</td>
      <td>40.717129</td>
      <td>-74.048751</td>
      <td>1766743528</td>
      <td>3.828116</td>
      <td>61.755647</td>
      <td>9.912088</td>
      <td>334.163523</td>
      <td>1006.264602</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-3702-5041</td>
      <td>14</td>
      <td>18.398797</td>
      <td>272.728448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a5c1a326-2f06-4c38-87ca-8692c07044ff</td>
      <td>0.58</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5b852930-c558-4ce1-8761-0fd6e0ae4344</td>
      <td>40.783002</td>
      <td>-73.976259</td>
      <td>1766743588</td>
      <td>11.679614</td>
      <td>56.133995</td>
      <td>1.348295</td>
      <td>157.370187</td>
      <td>1007.737435</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-4083-4722</td>
      <td>16</td>
      <td>16.130252</td>
      <td>238.168515</td>
    </tr>
  </tbody>
</table>
</div>



False negatives examples




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_id</th>
      <th>spoof_score_ml</th>
      <th>spoofed</th>
      <th>is_spoofed_ground_truth</th>
      <th>true_label</th>
      <th>installation_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>timestamp_unix</th>
      <th>horizontal_accuracy</th>
      <th>altitude</th>
      <th>speed</th>
      <th>bearing</th>
      <th>pressure_hpa</th>
      <th>mock_location_enabled</th>
      <th>device_is_charging</th>
      <th>wifi_bssid</th>
      <th>cell_tower_id</th>
      <th>num_satellites</th>
      <th>vertical_accuracy</th>
      <th>ambient_light_lux</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d1270765-5d39-4f6c-a63b-939d4cf13290</td>
      <td>0.29</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>05eb504c-b698-4e21-9ca3-57eff6740b4b</td>
      <td>40.793407</td>
      <td>-73.915567</td>
      <td>1766743728</td>
      <td>15.129599</td>
      <td>57.788534</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1003.484993</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-7596-8472</td>
      <td>12</td>
      <td>19.690095</td>
      <td>136.793347</td>
    </tr>
    <tr>
      <th>1</th>
      <td>efcbc1c7-19c4-4d4f-9f47-02999d905249</td>
      <td>0.06</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>6208ea36-df10-483a-a961-5d2c274e5f24</td>
      <td>40.703843</td>
      <td>-73.948827</td>
      <td>1766743408</td>
      <td>14.035643</td>
      <td>68.531940</td>
      <td>1.130479</td>
      <td>288.962041</td>
      <td>1004.068756</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-2908-6901</td>
      <td>11</td>
      <td>21.602901</td>
      <td>175.460312</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ffbd5d10-044f-4d6d-8939-b33a550fc8de</td>
      <td>0.12</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8c024d81-5488-40ee-865d-e0610ce03905</td>
      <td>40.690512</td>
      <td>-73.941630</td>
      <td>1766743598</td>
      <td>15.441016</td>
      <td>49.549634</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1008.260830</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>420-55-9636-3854</td>
      <td>17</td>
      <td>3.858546</td>
      <td>251.459559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fc7acd54-e048-4c5f-937b-a261ef6318ae</td>
      <td>0.23</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>698cbd1d-f647-4677-83e9-2a2598dcf13f</td>
      <td>40.735788</td>
      <td>-74.098360</td>
      <td>1766743668</td>
      <td>12.539873</td>
      <td>54.757683</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1005.875107</td>
      <td>False</td>
      <td>False</td>
      <td>0a:1b:2c:3d:4e:08</td>
      <td>420-55-4008-4997</td>
      <td>9</td>
      <td>19.447263</td>
      <td>176.366361</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161bcc86-e174-49e5-9f67-389ef62e78b5</td>
      <td>0.19</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>9e75f61c-1ac3-47dc-83fb-7e44241801b6</td>
      <td>40.790128</td>
      <td>-73.968532</td>
      <td>1766743558</td>
      <td>12.881011</td>
      <td>48.540646</td>
      <td>1.449734</td>
      <td>321.679358</td>
      <td>1005.663149</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>420-55-6851-2029</td>
      <td>18</td>
      <td>14.717809</td>
      <td>188.947594</td>
    </tr>
  </tbody>
</table>
</div>



### False positives patterns

- Low speed/indoor: 4 out of 5 events show speeds between 1.1 and 1.5m/s (walking) and all 6 have ambient_light_lux between 130 and 270 (indoor lighting)
- High satellite count: all have high satellite count and mock_location is false
- In indoor or slow-walking scenarios, sensor noise is minimal, so the model might be misinterpreting the clean signals as software simulated injection
- One event has significantly higher speed (9.9 m/s) and high score (0.46). This suggests the model might also be suspicious of rapid transitions with high precision

#### Tuning
- If more FPs occur at walking speed, we can use dynamic threshold: increase threshold when the speed is small (2m/s for instance)
- For clean signals: maybe add some engineered features like variance of ambient_light or pressure. If they fluctuates then more likely a real event.

### False negatives patterns
- 3 out of 5 events have a speed of 0.0 and a bearing of 0.0. The model likely treats stationary devices as "low risk." Attackers could exploit this by spoofing a fixed location.
- 2 remaining events show speeds of 1.1–1.4 m/s. These attackers are likely using spoofing apps that simulate a natural walking pace. Because the speed and bearing look realistic and the horizontal_accuracy is average.

#### Tuning
- Lowering the threshold means lower precision and hurting user experience with false alarms, 
- Instead we can add more features derived from sensor variance and environmental consistency. For example, cross check the local weather data with the GPS coordinates at corresponding timestamp to see pressure is consistent.


