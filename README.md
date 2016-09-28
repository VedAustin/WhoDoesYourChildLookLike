# WhoDoesYourChildLookLike
Does your child look more like Mom or Dad? Fear not, because a ML algorithm will give you an unbiased answer!

We all have been in situations where if you are a new parent (or an old parent), your friends and relatives ask if your child looks like you or your spouse. Lets put that to rest and use a simple machine learning algorithm to decide for you! 

The way it works is as follows:

1. Manually download a few dozen pics in which you all three are present.
2. Using `opencv` library, use their super cool face detection algorithm to crop only the faces. This is an important step if you want the face similarity to work only on the facial features and not on types of clothes, length of hair etc
3. Separate your cropped images into 3 separate folders.
4. Build a simple SVM model to train on Mom and Dad folders - this step is useful as it will automatically place the right face into the right folder when you have hundreds of images needed to train the full model.
5. Now download as many pics you have (100s) where the three of you are present - individually or in groups.
6. Run the SVM model to automatically put your pics into the right folder. Sometimes a few will slip and your pictures may end up in your spouse's folder. If this happens, manually put them in the right folder.
7. Now you have 3 folders with hopefully 100s of cropped images of Mom, Dad and Child (fewer of the child will suffice here)
8. Now it is time to train the full model. However you may not know the best algorithm to choose from. You could do manually or try something more automatic. For automatic evaluation use `tpot` - based on the results I ended up using a simpler logistic regression algorithm over SVM (my initial choice)
9. Test your Child face against the model and see who does it classify she most looks like. This is totally unscientific and goes completely against how one has been trained to use a classification model. But remember this is a fun project with a twist!

Based on the above, 90% of my child's pics look like my wife's. So unfortunately I will have to concede and conclude my child has not inherited much of her father's good looks ;)) Am I upset. Maybe .. just maybe!



