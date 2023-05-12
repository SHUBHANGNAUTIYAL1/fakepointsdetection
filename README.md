# fakepointsdetection

# Problem Statement
 
 
 Using the Aerial Lidar sensor 3D point cloud data is generated ( refer Fig 1), where each point represents the return of a laser pulse from a reflected surface, be it the ground or any off-terrain objects situated between the sensor and the ground. Common off-terrain objects include trees, crops, infrastructures etc. ​
 But importantly, airborne LiDAR systems provide comprehensive ground coverage, even under forest canopy. Reason being due to high sampling rate and relatively small laser footprint size, beams can often penetrate through the canopy to reach ground surface beneath. ​
 But under very dense forest canopies, we tend to get very few hits on ground. Thus the point density is drastically low compared to the non-dense forest. ​
 Additional Information which is not relevant to the scope of this problem, After removing outliers from the point cloud dataset, Ground-point classification automatically identifies each point as either a ground point or non-ground point. These Ground-points are further required to normalise point cloud data. But you don’t have to worry about normalisation. ​
 In case of non-dense canopies the normalisation process is smooth because of majority beams hitting the ground resulting in higher density of ground points for normalisation. ​
 But In Case of dense canopies where few beams hits the ground resulting in lower density of ground points thus the normalisation process gets confused and produces weird results .
 
 Our task :-
 
 The completed solution should be an algorithm that input a csv file, Identify void regions (polygon) where density of point cloud is below 500 pts/m2 [ variable ] 
 and save it as a csv. Then fill those void regions using some algorithm so that density > 500 pts/m2 [ variable ] 
 
 ![WhatsApp Image 2023-04-30 at 10 38 28](https://github.com/SHUBHANGNAUTIYAL1/fakepointsdetection/assets/79636717/88cff5dc-9df1-403b-ad73-0e99532b4b90)
