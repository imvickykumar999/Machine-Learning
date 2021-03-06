# Machine-Learning
- [Reference](https://www.pluralsight.com/guides/deep-learning-model-add)

----------------------------------

## Download Repository...
`git clone https://github.com/imvickykumar999/Machine-Learning.git`

---------------------------------

## Change Directory to "[adding ML](https://github.com/imvickykumar999/Machine-Learning/tree/main/adding%20ML)"
`C:\Users\Vicky\Desktop\Repository\Machine-Learning>cd "adding ML"`

--------------------------------------

## Training `python training.py 8`
- Here, 12 is epochs number's [./model/model_12.h5](https://github.com/imvickykumar999/Machine-Learning/blob/main/adding%20ML/model/model_12.h5)

`C:\Users\Vicky\Desktop\Repository\Machine-Learning\adding ML>python training.py 8`

    Epoch 1/8
    5000/5000 [==============================] - 6s 1ms/step - loss: 2407552.5000 - mae: 247.9589
    Epoch 2/8
    5000/5000 [==============================] - 5s 1ms/step - loss: 678.3096 - mae: 6.8089
    Epoch 3/8
    5000/5000 [==============================] - 4s 882us/step - loss: 2161.4172 - mae: 8.1126
    Epoch 4/8
    5000/5000 [==============================] - 5s 976us/step - loss: 907.1360 - mae: 6.3523
    Epoch 5/8
    5000/5000 [==============================] - 5s 967us/step - loss: 1490.4180 - mae: 8.4060
    Epoch 6/8
    5000/5000 [==============================] - 5s 1ms/step - loss: 1044.5524 - mae: 8.3311
    Epoch 7/8
    5000/5000 [==============================] - 4s 884us/step - loss: 847.4003 - mae: 7.3112
    Epoch 8/8
    5000/5000 [==============================] - 5s 911us/step - loss: 1252.2083 - mae: 7.7220
    63/63 [==============================] - 0s 509us/step - loss: 0.9051 - mae: 0.8891

    Test accuracy: 0.8890562653541565
    Test Loss: 0.9051101803779602

------------------------------------------

## Testing `python testing.py 100 200 12`
- sys.argv -> x y epochs

`C:\Users\Vicky\Desktop\Repository\Machine-Learning\adding ML>python testing.py 100 200 12`

    100.0 + 200.0 = 322.762939453125

    0 + 1 = 0.9196406006813049
    1 + 2 = 2.9276795387268066
    2 + 3 = 4.927828788757324
    3 + 4 = 6.927977561950684
    4 + 5 = 8.928126335144043
    5 + 6 = 10.928275108337402
    6 + 7 = 12.928425788879395
    7 + 8 = 14.928573608398438
    8 + 9 = 16.928726196289062
    9 + 10 = 18.92887306213379
    10 + 11 = 20.929019927978516
    11 + 12 = 22.929170608520508
    12 + 13 = 24.929323196411133
    13 + 14 = 26.92947006225586
    14 + 15 = 28.92961883544922
    15 + 16 = 30.929765701293945
    16 + 17 = 32.92991638183594
    17 + 18 = 34.93006134033203
    18 + 19 = 36.93021011352539
    19 + 20 = 38.93035888671875
    20 + 21 = 40.930511474609375
    21 + 22 = 42.93065643310547
    22 + 23 = 44.93080520629883
    23 + 24 = 46.93095016479492
    24 + 25 = 48.93109130859375
    25 + 26 = 50.93122482299805
    26 + 27 = 52.931365966796875
    27 + 28 = 54.93149948120117
    28 + 29 = 56.931640625
    29 + 30 = 58.93178176879883
    30 + 31 = 60.93191909790039
    31 + 32 = 62.93204879760742
    32 + 33 = 64.93219757080078
    33 + 34 = 66.93233489990234
    34 + 35 = 68.9324722290039
    35 + 36 = 70.93260955810547
    36 + 37 = 72.9327392578125
    37 + 38 = 74.9328842163086
    38 + 39 = 76.93302154541016
    39 + 40 = 78.93315124511719
    40 + 41 = 80.93330383300781
    41 + 42 = 82.93342590332031
    42 + 43 = 84.93356323242188
    43 + 44 = 86.93370819091797
    44 + 45 = 88.93385314941406
    45 + 46 = 90.93397521972656
    46 + 47 = 92.93412017822266
    47 + 48 = 94.93425750732422
    48 + 49 = 96.93438720703125
    49 + 50 = 98.93453216552734
    50 + 51 = 100.9346694946289
    51 + 52 = 102.934814453125
    52 + 53 = 104.9349365234375
    53 + 54 = 106.9350814819336
    54 + 55 = 108.93522644042969
    55 + 56 = 110.93535614013672
    56 + 57 = 112.93550109863281
    57 + 58 = 114.93563079833984
    58 + 59 = 116.93576049804688
    59 + 60 = 118.9359130859375
    60 + 61 = 120.93604278564453
    61 + 62 = 122.93618774414062
    62 + 63 = 124.93632507324219
    63 + 64 = 126.93647003173828
    64 + 65 = 128.9365997314453
    65 + 66 = 130.9367218017578
    66 + 67 = 132.9368896484375
    67 + 68 = 134.93701171875
    68 + 69 = 136.93714904785156
    69 + 70 = 138.93728637695312
    70 + 71 = 140.9374237060547
    71 + 72 = 142.9375762939453
    72 + 73 = 144.9376983642578
    73 + 74 = 146.93783569335938
    74 + 75 = 148.93798828125
    75 + 76 = 150.9381103515625
    76 + 77 = 152.93824768066406
    77 + 78 = 154.93838500976562
    78 + 79 = 156.93853759765625
    79 + 80 = 158.9386749267578
    80 + 81 = 160.9387969970703
    81 + 82 = 162.93894958496094
    82 + 83 = 164.9390869140625
    83 + 84 = 166.939208984375
    84 + 85 = 168.93934631347656
    85 + 86 = 170.93948364257812
    86 + 87 = 172.93963623046875
    87 + 88 = 174.93978881835938
    88 + 89 = 176.93991088867188
    89 + 90 = 178.94004821777344
    90 + 91 = 180.94015502929688
    91 + 92 = 182.9403076171875
    92 + 93 = 184.94046020507812
    93 + 94 = 186.94058227539062
    94 + 95 = 188.94073486328125
    95 + 96 = 190.94088745117188
    96 + 97 = 192.94100952148438
    97 + 98 = 194.94113159179688
    98 + 99 = 196.9412841796875
    99 + 100 = 198.94140625
    
----------------------------------------

## Graph : 
    y = x + (x+1)
    y = 2x + 1
    
----------------------------------

[![image](https://user-images.githubusercontent.com/50515418/148832627-4a556410-ffd9-4058-a471-8f1e7ad6a058.png)](https://github.com/imvickykumar999/Machine-Learning/blob/main/adding%20ML/Figure_1.png)

