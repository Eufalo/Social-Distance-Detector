class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		# list of lines that the objetc cross
		self.linecounted=[]
		# list of frame that the to cross one area detection
		self.framecounted=[]
		#flag to dont count the to over the line area
		self.flag_count=False
	
