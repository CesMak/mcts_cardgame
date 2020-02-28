# -*- coding: utf-8 -*-
"""
Created on Nov  21 2019
Changelog: 		27.11 Loading dics from file works and also saving these
					  Done: update_list
					  Done: setActorValues position
					  Done: move_stl position
					  Done: markiertes Element bekommt kurz Pose (bis reingeklickt)
					  Done: minipick Datei
					  Done: Build first gui that works!
TODO
		mach Pointer was geht da schief siehe addPointer
		mach 4 Dateien fuer die Sensoren!
		add Robot
@author: M. Lamprecht
"""
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QMainWindow, QFrame, QVBoxLayout, QListWidgetItem
from PyQt5 import uic, QtCore, QtGui
import sys
import vtk
from vtk import vtkCamera
import numpy as np
import math
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import ast
from xml.dom.minidom import parseString, parse
from json import loads, dumps


custom_dict = {} # stores all relevant values!
link_poses_actors = [] # stores all Pose actors (sind immer 6 enthalten daher nicht auslesbar)

class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
# see: https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html
	def __init__(self,parent=None):
		self.AddObserver("MiddleButtonPressEvent", self.middleButtonPressEvent)
		self.AddObserver("MiddleButtonReleaseEvent", self.middleButtonReleaseEvent)
		self.AddObserver("MouseWheelBackwardEvent", self.MouseWheelBackwardEvent)

	def middleButtonPressEvent(self,obj,event):
		#print("Middle Button pressed")
		self.OnMiddleButtonDown()
		return

	def middleButtonReleaseEvent(self,obj,event):
		#print("Middle Button released")
		self.OnMiddleButtonUp()
		return

	def MouseWheelBackwardEvent(self,obj,event):
		#print("Middle Button backward event")
		self.OnMouseWheelBackward()
		return


Ui_MainWindow, QtBaseClass = uic.loadUiType('mv_viewer.ui')
class mv_viewer(QMainWindow):

	def __init__(self):
		super(mv_viewer, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.saveConfig.clicked.connect(self.saveConfig)
		self.ui.openConfig.clicked.connect(self.openConfig)
		self.ui.pushButton_change_position.clicked.connect(self.move_stl)
		self.ui.comboBox_keys.currentTextChanged.connect(self.value_changed)
		self.ui.list_items.clicked.connect(self.setActorValues)
		self.ui.q1_slider.valueChanged.connect(self.q1_changed)
		self.ui.q2_slider.valueChanged.connect(self.q2_changed)
		self.ui.q3_slider.valueChanged.connect(self.q3_changed)
		self.ui.q4_slider.valueChanged.connect(self.q4_changed)
		self.ui.q5_slider.valueChanged.connect(self.q5_changed)
		self.ui.q6_slider.valueChanged.connect(self.q6_changed)

		self.vtkWidget = QVTKRenderWindowInteractor()
		layout = QVBoxLayout(self.ui.frame)
		layout.addWidget(self.vtkWidget)

		self.start_vtk()
		#self.minipick_setup()
		#self.update_list()
		#self.addDict_fromFile("MiniPick_json.txt")
		#self.test()
		#self.pip600_setup()
		#self.pip800_setup()
		#self.pip1200_setup()

		self.kr210()
		#self.robot_setup()


### Gui functions:
	def update_list(self):
		global custom_dict
		try:
			self.ui.list_items.clear()
			for k, v in custom_dict.items():
				item = QListWidgetItem()
				item.setText(v["Type"]+"\t"+v["name"]+"\t"+k)
				self.ui.list_items.addItem(item)
		except Exception as e:
			print("Error in update list"+str(e))

	def value_changed(self, value):
		global custom_dict
		item = self.ui.list_items.selectedItems()
		if len(item)==1:
			item = item[0]
			name=item.text().split("\t")
			# find the item in custom_dict
			dic = custom_dict[name[2]]
			if value in dic:
				self.ui.lineEdit_dict_value.setText(str(dic[value]))
			else:
				print("I do not know the value of key:", value)

	def setActorValues(self):
		global custom_dict
		item = self.ui.list_items.selectedItems()
		if len(item)==1:
			item = item[0]
			name=item.text().split("\t")
			# find the item in custom_dict
			dic = custom_dict[name[2]]

			if name[0] == "Camera":
				self.setCamera(dic)
			elif name[0] =="Robot":
				for key in dic:
					self.ui.comboBox_keys.addItem(key)
				return None
			else:
				self.ui.comboBox_keys.clear()
				for key in dic:
					self.ui.comboBox_keys.addItem(key)
				if "pos" in dic:
					self.ui.sp_x.setValue(float(dic["pos"][0]))
					self.ui.sp_y.setValue(float(dic["pos"][1]))
					self.ui.sp_z.setValue(float(dic["pos"][2]))
				else:
					print("Error sry position does not exist for", dic)
				if "rot" in dic:
					self.ui.sp_rx.setValue(float(dic["rot"][0]))
					self.ui.sp_ry.setValue(float(dic["rot"][1]))
					self.ui.sp_rz.setValue(float(dic["rot"][2]))
				else:
					print("Error sry rotation does not exist for", dic)

				#create a pose to highlight
				# when click in window help pose is removed again!
				pose1  = { "pos": dic["pos"], "rot": dic["rot"], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [800, 800, 800]}
				actor  = self.addPose(self.ren, pose1)

				#delete the help pose again
				self.ren.RemoveActor(actor)
				del custom_dict[self.getName_of_Actor(actor)]

		elif len(item) == 2:
			print("You selected 2 Items now I find the points of intersection!")
			line_item = item[0]
			stl_item  = item[1]
			line_item_name =line_item.text().split("\t")
			stl_item_name  =stl_item.text().split("\t")
			if (line_item_name[0] == "Cube") or (line_item_name[0] == "Point") or (line_item_name[0] == "Stl"):
				line_item = item[1]
				stl_item  = item[0]
				line_item_name =line_item.text().split("\t")
				stl_item_name  =stl_item.text().split("\t")
			self.check_intersection(line_item_name[0], custom_dict[line_item_name[2]], stl_item_name[0], custom_dict[stl_item_name[2]])


	def saveConfig(self):
		self.dictToFile(self.ui.file_name.text())
		print("I saved now to:", self.ui.file_name.text())

	def openConfig(self):
		self.addDict_fromFile(self.ui.file_name.text())
		self.update_list()

	def start_vtk(self):
		self.ren = vtk.vtkRenderer()
		self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
		self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

		self.iren.SetInteractorStyle(MyInteractorStyle(self.ren))
		self.iren.Initialize()
		self.iren.Start()
		#self.show()
		#self.ui.addWidget(self.vtkWidget)

#### VTK functions:
	def addRobot(self, renderer, dict):
		global custom_dict
		global link_poses_actors
		file_names = dict["file_names"]
		positions  = dict["positions"]
		names      = dict["names"]
		actor_names= []
		for i in range(0, len(file_names)):
			print("I added now:", file_names[i])
			current_dict   = {"name": names[i],"pos": positions[i],  "rot": [0, 0, 0], "opacity": 0.95,  "color": [1.0-i*0.1, 0.0+i*0.3, 0.5+i*0.1], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "file_name": file_names[i]}
			actor_names.append(self.addStl(renderer, current_dict))

			# add start link in origin:
			link = { "name": "l"+str(i), "pos": [0,0,0], "rot": [0.0, 0.0, 0.0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [0.2, 0.2, 0.2]}
			link_poses_actors.append(self.addPose(self.ren, link))

		# add Global Robot element with joints:
		joints = dict["joint_angles"]
		robot_dict = {"Type": "Robot", "name": dict["name"], "dh_table": dict["dh_table"], "file_names": file_names, "actor_names": actor_names}
		for i in range(0, len(joints)):
			robot_dict["q"+str(i)] = joints[i]
		custom_dict[dict["name"]] = robot_dict
		self.iren.GetRenderWindow().Render() # refresh window!

	def addStl(self, renderer, dict):
		global custom_dict
		reader = vtk.vtkSTLReader()
		reader.SetFileName(dict["file_name"])

		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([dict["rotation_3x3"], dict["pos"]])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)

		mapper = vtk.vtkPolyDataMapper()
		if vtk.VTK_MAJOR_VERSION <= 5:
			mapper.SetInput(reader.GetOutput())
		else:
			polydata = reader.GetOutputPort()
			mapper.SetInputConnection(polydata)

		actor = vtk.vtkActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Stl"
		custom_dict[self.getName_of_Actor(actor)] = dict

		actor.SetUserMatrix(transform.GetMatrix())
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		actor.SetMapper(mapper)
		renderer.AddActor(actor)
		self.iren.GetRenderWindow().Render() # refresh window!

	def addCube(self, renderer, dict):
		global custom_dict

		# create cube
		cube = vtk.vtkCubeSource()
		cube.SetXLength(dict["x_len"])
		cube.SetYLength(dict["y_len"])
		cube.SetZLength(dict["z_len"])

		# mapper
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(cube.GetOutputPort())

		# actor
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.SetPosition(dict["pos"])# stl is written at wrong position!
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		renderer.AddActor(actor)
		renderer.ResetCamera()

		# store values:
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Cube"
		custom_dict[self.getName_of_Actor(actor)] = dict

		self.iren.GetRenderWindow().Render() # refresh window!
		if dict["write_stl"]:
			 cube.SetCenter(dict["pos"])
			 stlWriter = vtk.vtkSTLWriter()
			 stlWriter.SetFileName(dict["file_name"])
			 stlWriter.SetInputConnection(cube.GetOutputPort())
			 stlWriter.Write()

	def addPoint(self, renderer, dict):
		global custom_dict
		point = vtk.vtkSphereSource()
		point.SetCenter(dict["pos"])
		point.SetRadius(dict["radius"])
		point.SetPhiResolution(dict["phi"])
		point.SetThetaResolution(dict["theta"])

		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(point.GetOutputPort())

		actor = vtk.vtkActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Point"
		custom_dict[self.getName_of_Actor(actor)] = dict
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])
		actor.GetProperty().BackfaceCullingOn()

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		renderer.AddActor(actor)
		renderer.ResetCamera()
		self.iren.GetRenderWindow().Render() # refresh window!

	def addText(self, renderer, dict):
		global custom_dict
		vtext = vtk.vtkVectorText()
		vtext.SetText(dict["text"])

		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(vtext.GetOutputPort())

		actor = vtk.vtkFollower()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Text"
		custom_dict[self.getName_of_Actor(actor)] = dict
		actor.SetMapper(mapper)
		actor.SetScale(dict["size"])
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])
		actor.AddPosition(dict["pos"][0]+10, dict["pos"][1]+10, dict["pos"][2]+10)

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		renderer.AddActor(actor)
		renderer.ResetCamera()
		self.iren.GetRenderWindow().Render() # refresh window!

	def addLine(self, renderer, dict):
		global custom_dict
		line = vtk.vtkLineSource()
		line.SetPoint1(dict["pos"])
		line.SetPoint2(dict["pos2"])

		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(line.GetOutputPort())

		actor = vtk.vtkActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Line"
		custom_dict[self.getName_of_Actor(actor)] = dict

		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])
		actor.GetProperty().SetLineWidth(dict["width"])

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		renderer.AddActor(actor)
		self.iren.GetRenderWindow().Render() # refresh window!

	def addCamera(self, dict):
		# this is a little bit different!
		# camera is not an actor!
		global custom_dict
		camera =vtkCamera()
		camera.SetPosition(dict["pos"]) #(958.7929232928253, -3650.878620373834, 3522.0295343192256)
		camera.SetFocalPoint(0, 0, 0)
		dict["Name"] = "Camera"+"_"+str(dict["Camera_nr"])
		dict["Type"]  = "Camera"
		custom_dict[dict["Name"]] = dict

	def addPointer(self, renderer, dict):
		global custom_dict
		pointer = vtk.vtkArrowSource()
		pointer.SetShaftRadius(dict["shaft_radius"]);
		pointer.SetShaftResolution(50)
		pointer.SetTipRadius(dict["tip_radius"]);
		pointer.SetTipLength(dict["tip_length"]);

		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(pointer.GetOutputPort())

		actor = vtk.vtkActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Pointer"
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])
		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])
		actor.AddPosition(dict["pos"][0], dict["pos"][1], dict["pos"][2])

		custom_dict[self.getName_of_Actor(actor)] = dict
		renderer.AddActor(actor)
		self.iren.GetRenderWindow().Render() # refresh window!

	def addPose(self, renderer, dict ):
		'''
		caution one 		actor = vtk.vtkAxesActor() add 6 actors(x_axis,y,z, x_text, y_text, z_text)
		'''
		global custom_dict
		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([dict["rotation_3x3"], dict["pos"]])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)
		transform.Scale(dict["arrow_length"])

		actor = vtk.vtkAxesActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Pose"
		custom_dict[self.getName_of_Actor(actor)] = dict

		actor.SetUserTransform(transform)
		# this sets the x axis label to red
		# axes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor(1,0,0);
		# the actual text of the axis label can be changed:
		actor.SetXAxisLabelText(dict["xlabel"]);
		actor.SetYAxisLabelText(dict["ylabel"]);
		actor.SetZAxisLabelText(dict["zlabel"]);

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		renderer.AddActor(actor)
		self.iren.GetRenderWindow().Render() # refresh window!
		return actor

	def addStl(self, renderer, dict):
		reader = vtk.vtkSTLReader()
		reader.SetFileName(dict["file_name"])

		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([dict["rotation_3x3"], dict["pos"]])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)

		mapper = vtk.vtkPolyDataMapper()
		if vtk.VTK_MAJOR_VERSION <= 5:
			mapper.SetInput(reader.GetOutput())
		else:
			polydata = reader.GetOutputPort()
			mapper.SetInputConnection(polydata)

		actor = vtk.vtkActor()
		dict["Name"] = self.getName_of_Actor(actor)
		dict["Type"]  = "Stl"
		custom_dict[self.getName_of_Actor(actor)] = dict

		actor.SetUserMatrix(transform.GetMatrix())
		actor.GetProperty().SetColor(dict["color"])
		actor.GetProperty().SetOpacity(dict["opacity"])

		actor.RotateX(dict["rot"][0])
		actor.RotateY(dict["rot"][1])
		actor.RotateZ(dict["rot"][2])

		actor.SetMapper(mapper)
		renderer.AddActor(actor)
		self.iren.Start()
		return self.getName_of_Actor(actor)

	def move_camera(self, zoom, azimuth):
		self.ren.GetActiveCamera().Zoom(zoom) #increase zoom
		self.ren.GetActiveCamera().Azimuth(azimuth) #increase zoom
#        self.ren.AddActor(actor)
#        self.iren.GetRenderWindow().Render() # refresh window!

	def getName_of_Actor(self, actor):
		split=str(actor).split("\n")
		return split[0]

	def getActor(self, name_of_actor):
		actors_collection = self.ren.GetActors()
		actors_collection.InitTraversal()
		for i in range(0, actors_collection.GetNumberOfItems()):
			next_actor = actors_collection.GetNextActor()
			if self.getName_of_Actor(next_actor) in name_of_actor:
				#print("I found your actor")
				return [next_actor, next_actor.GetProperty()]
		return None # <-- no actor found!

	def get_dic_of_actor(self, name_of_actor):
		global custom_dict
		counter = 0
		for i in custom_dict:
			if name_of_actor in i["Name"]:
				#print("I found your actor in custom_dict")
				return [i, counter]
			counter +=1

	def custom_add(self, name_to_add, content_dict):
			if name_to_add == "Cube":
				self.addCube(self.ren, content_dict)
			elif name_to_add == "Text":
				self.addText(self.ren, content_dict)
			elif name_to_add == "Pose":
				self.addPose(self.ren, content_dict)
			elif name_to_add == "Line":
				self.addLine(self.ren, content_dict)
			elif name_to_add == "Stl":
				self.addStl(self.ren, content_dict)
			elif name_to_add == "Point":
				self.addPoint(self.ren, content_dict)
			elif name_to_add == "Pointer":
				self.addPointer(self.ren, content_dict)
			elif name_to_add == "Camera":
				self.addCamera(content_dict)
			else:
				print("Error could not load", content_dict)

	def addDict_fromFile(self, file_name):
		# reads in from a dict filename!
		with open(file_name) as fd:
			doc = (fd.read())
		loaded_dict = (loads(doc))
		for k, v in loaded_dict.items():
			self.custom_add(v["Type"], v)

	def dictToFile(self, file_name):
		global custom_dict
		with open(file_name, "w") as text_file:
			text_file.write(dumps(custom_dict, indent=4, sort_keys=True))

	def setCamera(self, dict):
		print("Current Cam Position:", self.ren.GetActiveCamera().GetPosition(), "Cam Angle:", self.ren.GetActiveCamera().GetViewAngle())
		camera =vtkCamera()
		camera.SetPosition(dict["pos"])
		camera.SetFocalPoint(0, 0, 0)

		self.ren.SetActiveCamera(camera)
		istyle = self.iren.GetInteractorStyle()
		istyle.MouseWheelBackwardEvent(None, None) # update window!

	def setActorValues_sss(self, name_of_actor, type, key_, value_, pos=[0,0,0], rot=[0,0,0]):
		# 1.) get the actor from the list of all actors
		# 2.) find the properties of this actor in the custom_dict
		global custom_dict
		if type == "Robot":
			robotdict = custom_dict[name_of_actor]
			try:
				robotdict[key_] = (ast.literal_eval(value_))
			except Exception as e:
				print("inside: setActorValues_sss", e)
			self.changeRobotPose(robotdict)
			return None

		print("set actor values: Name of actor:,",  name_of_actor)
		[your_actor, prop] = self.getActor(name_of_actor)
		# remove this actor:
		#print("I try to remove the actor now.")
		self.ren.RemoveActor(your_actor)
		#print("Removed the actor!")

		# create a new actor with the properties in custom_dict
		dic = custom_dict[name_of_actor]
		# delete dic from list of dics
		dic["pos"] =  pos
		dic["rot"] = rot

		if ("Name" not in key_) and len(key_)>0 :
			dic[key_] = (ast.literal_eval(value_))
		del custom_dict[name_of_actor]

		# create this dic
		self.custom_add(dic["Type"], dic)

	def move_stl(self, val):
		x_pos = self.ui.sp_x.value()
		y_pos = self.ui.sp_y.value()
		z_pos = self.ui.sp_z.value()
		rx    = self.ui.sp_rx.value()
		ry    = self.ui.sp_ry.value()
		rz    = self.ui.sp_rz.value()
		key   = str(self.ui.comboBox_keys.currentText())
		value = self.ui.lineEdit_dict_value.text()

		item = self.ui.list_items.selectedItems()[0]
		name=item.text().split("\t")
		self.setActorValues_sss(name[2], name[0], key, value, pos=[x_pos, y_pos, z_pos], rot=[rx, ry, rz])
		self.iren.GetRenderWindow().Render() # refresh window!
		self.update_list()

#### additional Functions
	def loadSTL(self, filenameSTL):
		readerSTL = vtk.vtkSTLReader()
		readerSTL.SetFileName(filenameSTL)
		# 'update' the reader i.e. read the .stl file
		readerSTL.Update()

		polydata = readerSTL.GetOutput()

		# If there are no points in 'vtkPolyData' something went wrong
		if polydata.GetNumberOfPoints() == 0:
			raise ValueError(
				"No point data could be loaded from '" + filenameSTL)
			return None

		return polydata

	def move_camera(self, zoom, azimuth):
		self.ren.GetActiveCamera().Zoom(zoom) #increase zoom
		self.ren.GetActiveCamera().Azimuth(azimuth) #increase zoom

	def actor2dict(self, actor):
		# read out all information of an actor into a dict!
		split=str(actor).split("\n")

		## converts string to a dic!
		new_list = []
		for i in split:
			if ":" in i and not "none" in i:
				pos = i.find(":")
				i = i[:pos]+"\':"+"\'"+i[pos+1:]
				new_list.append(i)
		list_str = str(new_list).replace('"',"\'")
		list_str = list_str.replace("[","{")
		list_str = list_str.replace("]","}")
		list_str = list_str.replace(" ", "")

		dict = (ast.literal_eval(list_str))
		return dict


	def rotate_z(self, matrix_in, angle):
		 c, s = np.cos(self.deg2rad(angle)), np.sin(self.deg2rad(angle))
		 R = [[c, -s, 0], [s, c, 0], [0,0,1]]
		 return np.dot(matrix_in, R)

	def test(self):
		theta_i = [0, 				    0  ]
		d_i     = [0, 				    0  ]
		a_i     = [25, 			        480]
		alpha_i = [self.deg2rad(-90),   0, ]
		joints  = [0, 0] # joint angles in deg here!
		for i in range(0, len(joints)):
			joints[i] = self.deg2rad(joints[i])+theta_i[i]

		res = self.calc_dh(joints, d_i, a_i, alpha_i)
		T0i = []
		for jj in range(0, 2):
			m = np.round(self.calc_0Ti(res, i=jj), 3)
			T0i.append(m)
			print(m)

		T01 = T0i[0]# <-- drehung fuer Gelenk 2

		#print(rot)
		l1 = { "name": "l1", "pos": [25, 0, 0], "rot": [0.0, 0.0, 0.0], "rotation_3x3": [[1,0,0], [0,0,1], [0,-1,0]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l2 = { "name": "l2", "pos": [25, 0, 0], "rot": [0.0, 0.0, 0.0], "rotation_3x3": self.rotate_z([[1,0,0], [0,0,1], [0,-1,0]], 45), "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		self.addPose(self.ren, l1)
		self.addPose(self.ren, l2)


	def kr210(self):
		# Robot see: http://www.oemg.ac.at/Mathe-Brief/fba2015/VWA_Prutsch.pdf
		# for upright position:
		#		"positions": [[0, 0, 0], [-0.003, 0.001, 0.331], [.350, -0.037, 0.750], [0.350, -0.184, 2.000], [1.308, 0, 1.945], [1.850, 0, 1.945], [2.042, -0.000, 1.946]],
		#Denavit-Hartenberg
		theta_i = [0, 				    0					, -self.deg2rad(90), 0, 0, self.deg2rad(180)]
		d_i     = [0.675, 				0    				, 0.2				    , 1.6, 0, 0.190]
		a_i     = [0.35, 			    1.15				,-0.041			    ,0.0,  0, 0]
		alpha_i = [self.deg2rad(-90),   0					,   -self.deg2rad(90),  self.deg2rad(90),  self.deg2rad(-90), 0]

		bp     = "Robots\\kr210\\"
		robot  = {"name": "kr210", "joint_angles":[0, 0, 0, 0, 0, 0, 0], "dh_table":[theta_i, d_i, a_i, alpha_i],
		"names": ["base_link", "l1", "l2", "l3", "l4", "l5", "l6"],
		"positions": [[0, 0, 0], [-0.003, 0.001, 0.331], [.350, -0.037, 0.750], [0.350+1.15, -0.184, 0.75], [1.308+1.15, 0, 0.75-0.055], [1.850+1.15, 0, 0.75-0.055], [2.042+1.15, -0.000, 0.75-0.054]],
		"file_names": [bp+"base_linkx-9.stl", bp+"link_1_init.stl", bp+"link_2y9.stl", bp+"link_3_init.stl", bp+"link_4_initoy.stl", bp+"link_5_init.stl", bp+"link_6_init.stl"]}

		origin        = { "name": "origin", "pos": [0, 0, 0], "rot": [0, 0, 0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [10, 10, 10]}
		ground_plane  = { "name": "ground","pos": [0, 0, 0], "rot": [0, 0, 0],  "opacity": 1.0,   "x_len": 10,  "y_len": 10,  "z_len": 0.001, "color": [0.75, 0.75, 0.75], "write_stl": False, "file_name": "hallo.stl"}

		self.addRobot(self.ren, robot)
		self.addPose(self.ren, origin)
		self.addCube(self.ren, ground_plane)
		self.update_list()


	def robot_setup(self):
		# Robot
		#Denavit-Hartenberg
		theta_i = [0,		0, 				   self.deg2rad(-90), self.deg2rad(180), self.deg2rad(180), 0, 0 ]
		d_i     = [400, 	0, 				   0, 				  35, 				 420, 				0, 80]
		a_i     = [0, 		25, 			   0, 			      0,   				 0,  				0, 0 ]
		alpha_i = [0, 		self.deg2rad(-90),  0, 				  self.deg2rad(90),  self.deg2rad(90),  0, 0 ]

		bp     = "Robots\\kr6\\"
		robot  = {"name": "kr6", "joint_angles":[0, 0, 0, 0, 0, 0, 0], "dh_table":[theta_i, d_i, a_i, alpha_i],
		"names": ["base_link", "l1", "l2", "l3", "l4", "l5", "l6"],
		"positions": [[0,0,-400], [0,0,0], [25,0,0], [480,0,0], [480,0,35], [900,0,35], [980,0,35]],
		"file_names": [bp+"base_link.stl", bp+"link_1.stl", bp+"link_2_init.stl", bp+"link_3_init.stl", bp+"link_4_init.stl", bp+"link_5.stl", bp+"link_6.stl"]}

		origin        = { "name": "origin", "pos": [0, 0, 0], "rot": [0, 0, 0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [10000, 10000, 10000]}
		ground_plane  = { "name": "ground","pos": [0, 0, 0], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 12000,  "y_len": 12000,  "z_len": 0.1, "color": [0.75, 0.75, 0.75], "write_stl": False, "file_name": "hallo.stl"}

		self.addRobot(self.ren, robot)
		self.addPose(self.ren, origin)
		self.addCube(self.ren, ground_plane)
		self.update_list()

	def minipick_setup(self):

		text1         = { "name": "minipick_text", "pos": [800, 0, 600+150],  "rot": [0, 0, 0], "opacity": 0.7,   "text": "MINIPICK", "color": [1.0, 0.0, 0.0], "size": [60, 60, 60]}
		origin        = { "name": "origin", "pos": [0, 0, 0], "rot": [0, 0, 0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [10000, 10000, 10000]}
		ground_plane  = { "name": "ground","pos": [0, 0, 0], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 12000,  "y_len": 12000,  "z_len": 0.1, "color": [0.75, 0.75, 0.75], "write_stl": False, "file_name": "hallo.stl"}
		meas_volume   = { "name": "measvol","pos": [600, 0, 60+75], "rot": [0, 0, 0],  "opacity": 0.7,   "x_len": 300,  "y_len": 200,  "z_len": 150, "color": [0.75, 0.75, 0.75], "write_stl": False, "file_name": "hallo.stl"}
		minpick_stl   = { "name": "minpick_stl","pos": [600, 0, 150+600],  "rot": [0, 0, 0], "opacity": 1.0,  "color": [1.0, 1.0, 1.0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "file_name": "minipick_centered.stl"}
		minipick_pose = { "name": "minipick_pose","pos": [600, 0, 150+600], "rot": [0, 0, 0], "rotation_3x3": [[1,0,0], [0,1,0], [0,0,1]], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [80, 80, 80]}
		# distance to sensor=450 --> x=300, y=196
		dis_1 		  = { "name": "top","pos": [600, 0, 200], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 300,  "y_len": 196,  "z_len": 6, "color": [0.75, 1.0, 0.24], "write_stl": False, "file_name": "hallo.stl"} # cube
		# distance to sensor=500 --> x=367, y=235
		dis_2 		  = { "name": "mid1","pos": [600, 0, 150], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 367,  "y_len": 235,  "z_len": 6, "color": [0.75, 1.0, 0.24], "write_stl": False, "file_name": "hallo.stl"} # cube
		# distance to sensor=575 --> x=425, y=273
		dis_3 		  = { "name": "mid2","pos": [600, 0, 75], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 425,  "y_len": 273,  "z_len": 6, "color": [0.75, 1.0, 0.24], "write_stl": False, "file_name": "hallo.stl"} # cube
		# distance to sensor=650 --> x=459, y=300
		dis_4 		  = { "name": "bot", "pos": [600, 0, 0], "rot": [0, 0, 0],  "opacity": 0.25,   "x_len": 459,  "y_len": 300,  "z_len": 6, "color": [0.75, 1.0, 0.24], "write_stl": False, "file_name": "hallo.stl"} # cube

		# Sichtstrahlen:
		line1  = { "name": "l1",  "pos": [600, 0, 150+600],  "rot": [0, 0, 0], "opacity": 0.5,   "pos2": [600-459/2.0, 150, 0], "color": [0.0, 1.0, 0.0], "size": [10, 10, 10], "width": 1}
		line2  = { "name": "l2",  "pos": [600, 0, 150+600],  "rot": [0, 0, 0], "opacity": 0.5,   "pos2": [600-459/2.0, -150, 0], "color": [0.0, 1.0, 0.0], "size": [10, 10, 10], "width": 1}
		line3  = { "name": "l3",  "pos": [600, 0, 150+600],  "rot": [0, 0, 0], "opacity": 0.5,   "pos2": [600+459/2.0, -150, 0], "color": [0.0, 1.0, 0.0], "size": [10, 10, 10], "width": 1}
		line4  = { "name": "l4",  "pos": [600, 0, 150+600],  "rot": [0, 0, 0], "opacity": 0.5,   "pos2": [600+459/2.0, 150, 0], "color": [0.0, 1.0, 0.0], "size": [10, 10, 10], "width": 1}

		# camera:
		cam1  = { "name": "Cam1",  "pos": [958.7929232928253, -3650.878620373834, 3522.0295343192256], "Camera_nr": 1}

		# pointer:
		pointer  = { "name": "pp",  "pos": [0, 0, 0],  "rot": [0, 0, 0], "shaft_radius": 0.015*10000000,   "tip_radius": 0.05*100000, "tip_length": 0.1*1000000, "color": [0.0, 1.0, 0.0], "opacity": 0.5,}

		# Robot
		#Denavit-Hartenberg
		theta_i = [	0, 				   self.deg2rad(-90), self.deg2rad(180), self.deg2rad(180), 0, 0 ]
		d_i     = [	400, 				0, 				  35, 				 420, 				0, 80]
		a_i     = [ 25, 			    0, 			      0,   				 0,  				0, 0 ]
		alpha_i = [ self.deg2rad(-90),  0, 				  self.deg2rad(90),  self.deg2rad(90),  0, 0 ]

		bp     = "Robots\\kr6\\"
		robot  = {"name": "kr6", "joint_angles":[0, 0, 0, 0, 0, 0, 0], "dh_table":[theta_i, d_i, a_i, alpha_i], "names": ["base_link", "l1", "l2", "l3", "l4", "l5", "l6"], "positions": [[0,0,0], [0,0,400], [25,0,400], [480,0,400], [480,0,435], [900,0,435], [980,0,435]], "file_names": [bp+"base_link.stl", bp+"link_1.stl", bp+"link_2.stl", bp+"link_3.stl", bp+"link_4.stl", bp+"link_5.stl", bp+"link_6.stl"]}

		for i in [line1, line2, line3, line4]:
			self.addLine(self.ren, i)
		for i in [ground_plane, meas_volume, dis_1, dis_2, dis_3, dis_4]:
			self.addCube(self.ren, i)
		for i in [minipick_pose, origin]:
			self.addPose(self.ren, i)
		self.addText(self.ren, text1)
		self.addStl(self.ren, minpick_stl)

		self.addCamera(cam1)
		self.addRobot(self.ren, robot)
		#self.addPointer(self.ren, pointer)

		#save as json:
		self.dictToFile("MiniPick_json.txt")
		self.update_list()


	def pip600_setup(self):
		self.addText(self.ren, [1800, 0, 1510], "PIP600", size=60, color=[1.0, 0, 0])

		self.add_cube(self.ren, 600, 400, 400, x_pos=1600, z_pos=10+200, opacity=0.7) # measurement volume
		self.read_in_stl(self.ren, filename="PP600_centered.stl", position=[1600, 0, 1510], rotation=[[1,0,0], [0,1,0], [0,0,1]])
		self.addPose(self.ren, [1600, 0, 1510], [[1,0,0], [0,1,0], [0,0,1]]) # origin

		# distance to sensor=1510 --> x=836, y=584
		self.add_cube(self.ren, 836, 584, 6, x_pos=1600, z_pos=0, color=[0.75, 1.0, 0.24])
		#Lines
		self.addLine(self.ren, [1510, 0, 1600], [1600-836/2, 584/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [1510, 0, 1600], [1600+836/2, 584/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [1510, 0, 1600], [1600-836/2, -584/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [1510, 0, 1600], [1600+836/2, -584/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)

	def pip800_setup(self):
		self.addText(self.ren, [2800, 0, 1900], "PIP800", size=120, color=[1.0, 0, 0])

		self.add_cube(self.ren, 800, 600, 600, x_pos=2800, z_pos=300, opacity=0.7) # measurement volume
		self.read_in_stl(self.ren, filename="PP800_centered.stl", position=[2800, 0, 1900], rotation=[[1,0,0], [0,1,0], [0,0,1]])
		self.addPose(self.ren, [2800, 0, 1900], [[1,0,0], [0,1,0], [0,0,1]]) # origin

		# distance to sensor=1300 --> x=883 y=662
		self.add_cube(self.ren, 883, 662, 6, x_pos=2800, z_pos=600, color=[0.75, 1.0, 0.24], file_name="pip800_top.stl", write_stl=1)

		self.add_cube(self.ren, 1062, 810, 6, x_pos=2800, z_pos=300, color=[0.75, 1.0, 0.24])
		self.add_cube(self.ren, 1242, 949, 6, x_pos=2800, z_pos=0, color=[0.75, 1.0, 0.24])

		#Lines
		self.addLine(self.ren, [2800, 0, 1600], [2800-1242/2, -949/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [2800, 0, 1600], [2800-1242/2, +949/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [2800, 0, 1600], [2800+1242/2,-949/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [2800, 0, 1600], [2800+1242/2, 949/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)


	def pip1200_setup(self):
		self.addText(self.ren, [5000, 0, 3200], "PIP1200", size=200, color=[1.0, 0, 0])

		self.add_cube(self.ren, 1200, 1000, 1000, x_pos=5000, z_pos=500, opacity=0.7) # measurement volume
		self.read_in_stl(self.ren, filename="PP1200_centered.stl", position=[5000, 0, 3200], rotation=[[1,0,0], [0,1,0], [0,0,1]])
		self.addPose(self.ren, [5000, 0, 3200], [[1,0,0], [0,1,0], [0,0,1]]) # origin

		self.add_cube(self.ren, 1330, 1130, 6, x_pos=5000, z_pos=1000, color=[0.75, 1.0, 0.24])
		self.add_cube(self.ren, 1570, 1300, 6, x_pos=5000, z_pos=500, color=[0.75, 1.0, 0.24])
		self.add_cube(self.ren, 1830, 1480, 6, x_pos=5000, z_pos=0, color=[0.75, 1.0, 0.24])

		#Lines
		self.addLine(self.ren, [5000, 0, 3200], [5000-1830/2, -1480/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [5000, 0, 3200], [5000+1830/2, -1480/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [5000, 0, 3200], [5000-1830/2, +1480/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)
		self.addLine(self.ren, [5000, 0, 3200], [5000+1830/2, +1480/2, 0], color=[0.0, 1.0, 0.0], opacity=0.5)

	def write_stl(self, file_name, output_port):
		 stlWriter = vtk.vtkSTLWriter()
		 stlWriter.SetFileName(file_name)
		 stlWriter.SetInputConnection(output_port)
		 stlWriter.Write()
		 print("I wrote the file:", file_name)

	def create_stl_of_type(self, dict):
		if dict["Type"]=="Cube":
			cube = vtk.vtkCubeSource()
			cube.SetXLength(dict["x_len"])
			cube.SetYLength(dict["y_len"])
			cube.SetZLength(dict["z_len"])
			cube.SetCenter(dict["pos"])
			self.write_stl(dict["file_name"], cube.GetOutputPort())
		elif dict["Type"]=="Point":
			point = vtk.vtkSphereSource()
			point.SetCenter(dict["pos"])
			point.SetRadius(dict["radius"])
			point.SetPhiResolution(dict["phi"])
			point.SetThetaResolution(dict["theta"])
			self.write_stl(dict["file_name"], point.GetOutputPort())
		else:
			print("error in create_stl_of_type")

	def check_intersection(self, line_type, line_dic, stl_type, stl_dict):
		# seee: https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/
		if not line_type == "Line":
			print(line_type, "is not a line:")
			print(line_dic)
			return
		stl_name = ""
		if (stl_type=="Cube") or (stl_type == "Point"):
			print("I create the stl now!")
			self.create_stl_of_type(stl_dict)
			stl_name = stl_dict["file_name"]
		elif stl_type=="Stl":
			print("Is a stl already")
			stl_name = stl_dict["file_name"]
		else:
			print(stl_type, "is not a stl, cube or point:")
			print(stl_dict)
			return

		mesh = self.loadSTL(stl_name)
		obbTree = vtk.vtkOBBTree()
		obbTree.SetDataSet(mesh)
		obbTree.BuildLocator()
		pointsVTKintersection = vtk.vtkPoints()
		code = obbTree.IntersectWithLine(line_dic["pos"], line_dic["pos2"], pointsVTKintersection, None)

		pointsVTKIntersectionData = pointsVTKintersection.GetData()
		noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

		pointsIntersection = []
		for idx in range(noPointsVTKIntersection):
			_tup = pointsVTKIntersectionData.GetTuple3(idx)
			pointsIntersection.append(_tup)

		print("Intersection_points:", pointsIntersection)
		print("code:", code)

		for p in pointsIntersection:
			point = { "name": "Intersec", "pos": p,  "rot": [0, 0, 0], "opacity": 0.8,   "radius": 50, "color": [1.0, 0.0, 0.0], "phi": 0, "theta": 0}
			self.addPoint(self.ren, point)
			self.update_list()
			print("Caution MV is not fully visible!", p)
		print("end of check intersection")
#
#
# Robot Transform
#
#
	def get_dh_transform(self, angle_joint, alpha, an, dn):
		 return np.array([[np.cos(angle_joint),-np.sin(angle_joint)*np.cos(alpha),np.sin(angle_joint)*np.sin(alpha),an*np.cos(angle_joint)],
						[np.sin(angle_joint)  ,np.cos(angle_joint)*np.cos(alpha),-np.cos(angle_joint)*np.sin(alpha),an*np.sin(angle_joint)],
						[0,					   np.sin(alpha)					,np.cos(alpha)					   ,dn],
						[0,0,0,1]])

	def calc_dh(self, theta, d, a, alpha):
		res = []
		for i in range(0, len(theta)):
			curr = self.get_dh_transform(theta[i], alpha[i], a[i], d[i])
			res.append(curr)
		return res

	def calc_0Ti(self, res, number=0, i=15):
		if len(res)>1 and number<i:
			tmp1 = res[0]
			tmp2 = res[1]
			del res[1]
			res[0]=np.dot(tmp1, tmp2)
			return (self.calc_0Ti(res, number=number+1, i=i))
		else:
			return res[0]

	def updatePose(self, your_actor, rot, pos):
		# TODO update also custom dict!!!
		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([rot, pos])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)

		your_actor.SetUserMatrix(transform.GetMatrix())
		self.iren.GetRenderWindow().Render() # refresh window!
		self.update_list()

	def updateJoint(self, link_pos, robot_dic, rot, pos):
		global custom_dict
		name_of_actor = robot_dic["actor_names"][link_pos]
		[your_actor, prop] = self.getActor(name_of_actor)

		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([rot, pos])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)

		your_actor.SetUserMatrix(transform.GetMatrix())
		self.iren.GetRenderWindow().Render() # refresh window!


	def setJoint(self, link_pos, robot_dic, rot, pos):
		print("inside setJoint")
		global custom_dict
		name_of_actor = robot_dic["actor_names"][link_pos]
		print("actor name:", name_of_actor)
		[your_actor, prop] = self.getActor(name_of_actor)
		self.ren.RemoveActor(your_actor)
		dic = custom_dict[name_of_actor]

		# delete dic from list of dics
		dic["pos"] =  pos
		dic["rotation_3x3"] = rot
		del custom_dict[name_of_actor]

		print(dic)

		# create this dic
		self.custom_add(dic["Type"], dic)


	def apply_joints(self):
		global link_poses_actors
		# joint multiplication (if angles are negative)
		joints_multiplication = [1, 1, 1, -1, 1, -1]#TODO this in dict
		origin_position       = [0, 0, 0] #TODO this in dict

		joints  = [self.ui.q1_slider.value(), self.ui.q2_slider.value(), self.ui.q3_slider.value(), self.ui.q4_slider.value(),  self.ui.q5_slider.value(), self.ui.q6_slider.value()]
		[dict, key] = self.getDictbyName("Robot", "kr210")

		theta_i = [0, 				    0					, -self.deg2rad(90), 0, 0, self.deg2rad(180)]
		d_i     = [0.675, 				0    				, 0.2				    , 1.6, 0, 0.190]
		a_i     = [0.35, 			    1.15				,-0.041			    ,0.0,  0, 0]
		alpha_i = [self.deg2rad(-90),   0					,   -self.deg2rad(90),  self.deg2rad(90),  self.deg2rad(-90), 0]

		# joint angles in deg here!
		for ppp in range(0, len(theta_i)):
			joints[ppp] = joints_multiplication[ppp]*self.deg2rad(joints[ppp])+theta_i[ppp]
		res = self.calc_dh(joints, d_i, a_i, alpha_i)
		T0i = []
		for jj in range(0, len(theta_i)):
			res = self.calc_dh(joints, d_i, a_i, alpha_i) # TODO res macht was komisches!!! durch del befehl! (daher hier in for loop!)
			m = np.round(self.calc_0Ti(res, i=jj), 8)
			T0i.append(m)

		for i in range(0, len(T0i)):
			curr_pose = T0i[i]
			if i == 0: # origin (with origin position!)
				link = { "pos": origin_position,  "rotation_3x3":   curr_pose[0:3,0:3]}
				self.updateJoint(1, dict, curr_pose[0:3,0:3], origin_position)
			else:
				prev_pose = T0i[i-1]
				if i == len(T0i)-1: # last link!
					self.updateJoint(i+1, dict, curr_pose[0:3,0:3], curr_pose[0:3, 3])
					link = {"pos": curr_pose[0:3, 3], "rotation_3x3": curr_pose[0:3,0:3]}
				else:
					self.updateJoint(i+1, dict, curr_pose[0:3,0:3], prev_pose[0:3, 3])
					link = { "name": "l"+str(i), "pos": prev_pose[0:3, 3], "rotation_3x3": curr_pose[0:3,0:3]}
			self.updatePose(link_poses_actors[i], link["rotation_3x3"], link["pos"])
		self.update_list()

	def apply_joints_kr6(self):
		joints  = [self.ui.q1_slider.value(), self.ui.q2_slider.value(), self.ui.q3_slider.value(), self.ui.q4_slider.value(),  self.ui.q5_slider.value()]
		[dict, key] = self.getDictbyName("Robot", "kr6")

		# joint5 has an offset!!! (how to handle this?!)
		theta_i = [0, 				    0					, self.deg2rad(90), self.deg2rad(90), 0]
		d_i     = [0, 				    0    				, 0				  , 400, 0]
		a_i     = [25, 			        480					,			 -35			      ,0, 0]
		alpha_i = [self.deg2rad(-90),   0					,   self.deg2rad(90),  self.deg2rad(-180), 0]

		 # joint angles in deg here!
		for ppp in range(0, len(theta_i)):
			joints[ppp] = self.deg2rad(joints[ppp])+theta_i[ppp]

		res = self.calc_dh(joints, d_i, a_i, alpha_i)
		T0i = []
		for jj in range(0, len(theta_i)):
			res = self.calc_dh(joints, d_i, a_i, alpha_i) # TODO res macht was komisches!!! durch del befehl! (daher hier in for loop!)
			print("Laenge res", len(res))
			m = np.round(self.calc_0Ti(res, i=jj), 8)
			T0i.append(m)
			print(jj)
			print(m)

		T01 = T0i[0]# <-- drehung fuer Gelenk 2
		T02 = T0i[1]
		T03 = T0i[2]
		T04 = T0i[3]
		T05 = T0i[4]
		print("T03")
		print(T03)

		# print("T01, self")
		# T01_self = np.dot(res[0], res[1])
		# print(T01_self)
		# print("T02, self")
		# T02_self = np.dot(T01_self, res[2])
		# print(T02_self)
		# print("T03, self")
		# T03_self = np.dot(T02_self, res[3])
		# print(T03_self)

		#T04 = T0i[3]
		#T05 = T0i[4]
		#self.write_custom_dh_stl(theta_i, d_i, a_i, alpha_i)
		print("Euler: Initial Drehung Teil 2:", self.rot2eul(T02[0:3,0:3]))
		print("Euler: Initial Drehung Teil 3:", self.rot2eul(T03[0:3,0:3]))

		l0 = { "name": "l0", "pos": [0, 0, 0], "rot": [0.0, 0.0, 0.0], "rotation_3x3": self.rotate_z(np.eye(3), self.ui.q1_slider.value()), "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l1 = { "name": "l1", "pos": T01[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T02[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l2 = { "name": "l2", "pos": T02[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T03[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l3 = { "name": "l3", "pos": T03[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T04[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l4 = { "name": "l4", "pos": T04[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T05[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}

		self.addPose(self.ren, l0)
		self.addPose(self.ren, l1)
		self.addPose(self.ren, l2)
		self.addPose(self.ren, l3)
		self.addPose(self.ren, l4)


		self.updateJoint(1, dict, self.rotate_z(np.eye(3), self.ui.q1_slider.value()), [0, 0, 0])
		self.updateJoint(2, dict, T02[0:3,0:3], T01[0:3, 3])
		self.updateJoint(3, dict, T03[0:3,0:3], T02[0:3, 3])
		self.updateJoint(4, dict, T04[0:3,0:3], T03[0:3, 3])
		self.update_list()

	def write_custom_dh_stl(self, theta_i, d_i, a_i, alpha_i):
		res = self.calc_dh(theta_i, d_i, a_i, alpha_i)
		T0i = []
		for jj in range(0, len(theta_i)):
			res = self.calc_dh(theta_i, d_i, a_i, alpha_i) # TODO res macht was komisches!!! durch del befehl! (daher hier in for loop!)
			m = np.round(self.calc_0Ti(res, i=jj), 8)
			T0i.append(m)

		tmp = 0
		for j in T0i:
			rot = j[0:3,0:3]
			print("Euler: Initial Drehung Teil "+str(tmp), self.rot2eul(rot))
			tmp+=1

		# create a transform that rotates the cone
		#For a Robot offset here!

		T02 = T0i[0]# <-- für link3.stl T0i[2]

		reader = vtk.vtkSTLReader()
		reader.SetFileName("Robots\\kr210\\link_1.stl")
		#reader.SetCenter(0,0,10)
		#vtkTransformPolyDataFilter
		# --> see: http://vtk.1045678.n5.nabble.com/vtkTransform-problem-td1243732.html
		# tf = vtk.vtkTransformPolyDataFilter()
		# tf.SetInputConnection(reader.GetOutputPort())


		vtk_matrix = vtk.vtkMatrix4x4()
		motion_matrix = np.column_stack([T02[0:3,0:3], [0,0,0]])
		motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
		for i in range(4):
			for j in range(4):
				vtk_matrix.SetElement(i, j, motion_matrix[i,j])
		transform = vtk.vtkTransform()
		transform.Concatenate(vtk_matrix)

		transform_robot = vtk.vtkTransform()
		transform_robot.RotateWXYZ(270, 0, 1, 0)
		transformFilter=vtk.vtkTransformPolyDataFilter()
		transformFilter.SetTransform(transform)
		transformFilter.SetInputConnection(reader.GetOutputPort())
		transformFilter.Update()

		stlWriter = vtk.vtkSTLWriter()
		stlWriter.SetFileName("Robots\\kr210\\link_1_init.stl")
		stlWriter.SetInputConnection(transformFilter.GetOutputPort())
		stlWriter.Write()

	def q6_changed(self, value):
		self.apply_joints()

	def q5_changed(self, value):
		self.apply_joints()

	def q4_changed(self, value):
		self.apply_joints()

	def q3_changed(self, value):
		self.apply_joints()

	def q2_changed(self, value):
		self.apply_joints()


	def q1_changed(self, value):
		self.apply_joints()

	def getDictbyName(self, type, name):
		global custom_dict
		for key in custom_dict:
			my_sub_dic = custom_dict[key]
			for i in my_sub_dic:
					if i=="name" and my_sub_dic[i] == name:
						if my_sub_dic["Type"] == type:
							return [my_sub_dic, key]
		return [None, None]

	def changeRobotPose(self, dict):
		global custom_dict
		print("inside change RobotPose")
		print(dict)
		[theta_i, d_i, a_i, alpha_i] =  dict["dh_table"]
		joints = [ dict["q0"], dict["q1"], dict["q2"], dict["q3"], dict["q4"], dict["q5"], dict["q6"]]
		print("joints: ", joints)
		print("theta ", theta_i)
		print("d_i   ", d_i)
		print("a_i   ", a_i)
		print("alpha ", alpha_i)
		for i in range(0, len(theta_i)):
			theta_i[i] += self.deg2rad(joints[i])

		res = self.calc_dh(theta_i, d_i, a_i, alpha_i)
		T01 = self.calc_0Ti(res, i=0)
		T02 = np.round(self.calc_0Ti(res, i=1),2) ## ist das gleiche! np.round(np.dot(res[0], res[1]))

		#T02 = np.linalg.inv(T02)

		# T03 = self.calc_0Ti(res, i=3)
		# T04 = self.calc_0Ti(res, i=4)
		# T05 = self.calc_0Ti(res, i=5)
		#T06 = self.calc_0Ti(res, i=6)

		print("T01:")
		print(T01)

		print("T02:")
		print(T02)

		print("extracted: ")
		rotation1 = T01[0:3,0:3]
		trans1    = T01[0:3, 3]
		print(rotation1)
		print("trans:", trans1)
		print("Rotation2:")
		print(T02[0:3,0:3])
		print(self.rot2eul(T02[0:3,0:3]))


		rot2 = np.linalg.inv(T02[0:3,0:3])

		# add pose here correctly
		l1 = { "name": "l1", "pos": trans1, "rot": [0.0, 0.0, 0.0], "rotation_3x3": rotation1, "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		l2 = { "name": "l2", "pos":  T02[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": rot2, "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# l3 = { "name": "l1", "pos":  T03[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T03[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# l4 = { "name": "l1", "pos":  T04[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T04[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# l5 = { "name": "l1", "pos":  T05[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T05[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		self.addPose(self.ren, l1)
		self.addPose(self.ren, l2)
		# self.addPose(self.ren, l3)
		# self.addPose(self.ren, l4)
		# self.addPose(self.ren, l5)
		# print("T02:")
		# print(T02)
		# print("finished here")
		# # get numbers here!

		# change joints:
		self.setJoint(1, dict, rotation1, trans1)
		self.setJoint(2, dict, rot2,  T02[0:3, 3])

		return None
		#
		# pprint((self.calculate0TE(res)))


		# theta_i = [0,		0, 				   self.deg2rad(-90), self.deg2rad(180), self.deg2rad(180), 0, 0 ]
		# d_i     = [400, 	0, 				   0, 				  35, 				 420, 				0, 80]
		# a_i     = [0, 		25, 			   0, 			      0,   				 0,  				0, 0 ]
		# alpha_i = [0, 		self.deg2rad(-90), 0, 				  self.deg2rad(90),  self.deg2rad(90),  0, 0 ]


		#1. get joints
		#2. get DH - Table
		#3.

	def rot2eul(self, R):
	    beta = -np.arcsin(R[2,0])
	    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
	    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
	    return np.array((self.rad2deg(alpha), self.rad2deg(beta), self.rad2deg((gamma))))

	def deg2rad(self,angle):
		radangle=angle*np.pi/180
		return radangle

	def rad2deg(self, angle):
		return angle*180/np.pi
#
#
# End of Robot Transform
#
#
	def euler_angles_from_rotation_matrix(self, roll, pitch, yaw):
		yawMatrix = np.matrix([
		[math.cos(yaw), -math.sin(yaw), 0],
		[math.sin(yaw), math.cos(yaw), 0],
		[0, 0, 1]
		])

		pitchMatrix = np.matrix([
		[math.cos(pitch), 0, math.sin(pitch)],
		[0, 1, 0],
		[-math.sin(pitch), 0, math.cos(pitch)]
		])

		rollMatrix = np.matrix([
		[1, 0, 0],
		[0, math.cos(roll), -math.sin(roll)],
		[0, math.sin(roll), math.cos(roll)]
		])

		R = yawMatrix * pitchMatrix * rollMatrix
		return R

if __name__ == '__main__':
	app = QApplication(sys.argv)
	main = mv_viewer()
	main.show()
	sys.exit(app.exec_())


#########
#########
######### NOTES
#########
#########

#from pycaster import pycaster
##### installing vtk
### see mey answer on: https://stackoverflow.com/questions/19019720/importerror-dll-load-failed-1-is-not-a-valid-win32-application-but-the-dlls/58989774#58989774
#pip install python-vtk  -> did not work to install.
# worked to install but py -m pip install --user vtk cannot import it!
# I am using: Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
# get it from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/
# pip install VTK-8.2.0-cp37-cp37m-win32.whl
# pip install pycaster  ## -> stl schneiden etc. intersection points etc. https://pypi.org/project/pycaster/


## Examples see here: https://vtk.org/Wiki/VTK/Examples/Python
## https://vtk.org/Wiki/VTK/Examples/Python/Interaction/HighlightAPickedActor
## https://vtk.org/Wiki/VTK/Examples/Python/Animation
## https://lorensen.github.io/VTKExamples/site/Python/

## Description of KUKA robots:
# KUKA LWR 4+
#https://github.com/epfl-lasa/kuka-lwr-ros/tree/master/kuka_lwr/lwr_description/meshes
#KR3, KR5, KR6, KR10, KR120, KR210
#https://github.com/ros-industrial/kuka_experimental
# -> wir benoetigen einen KR60 Ha und einen Agilus: KR6 R900_sixx
# siehe: https://github.com/ros-industrial/kuka_experimental/blob/indigo-devel/kuka_kr6_support/urdf/kr6r900sixx_macro.xacro

# Generating exe problems:
#https://github.com/pyinstaller/pyinstaller/issues/2129
# in spec file: hiddenimports = ['vtkmodules','vtkmodules.all','vtkmodules.qt.QVTKRenderWindowInteractor','vtkmodules.util','vtkmodules.util.numpy_support']



# in move_stl
# vtk_matrix = vtk.vtkMatrix4x4()
# motion_matrix = np.column_stack([ self.euler_angles_from_rotation_matrix(rx_pos, ry_pos, rz_pos), [x_pos, y_pos, z_pos]])
# motion_matrix = np.row_stack([motion_matrix, np.asfarray([0,0,0,1])])
# for i in range(4):
#   for j in range(4):
#       vtk_matrix.SetElement(i, j, motion_matrix[i,j])
# transform = vtk.vtkTransform()
# transform.Concatenate(vtk_matrix)
#actor.SetUserMatrix(transform.GetMatrix())



# save this dict to xml:
# xml = parseString(dicttoxml(custom_dict, custom_root='MiniPickxml', attr_type=True))
# xml_to_print = (xml.toprettyxml())
#
# with open("MiniPickxml.xml", "w") as text_file:
# 	text_file.write(xml_to_print)
# print("I wrote your xml file")



#### Working example:
# class MainWindow(QMainWindow):
#
#   def __init__(self, parent = None):
#       QMainWindow.__init__(self, parent)
#
#       self.frame = QFrame()
#       self.vl = QVBoxLayout()
#       self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
#       self.vl.addWidget(self.vtkWidget)
#
#       self.ren = vtk.vtkRenderer()
#       self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
#       self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
#
#       # Create source
#       source = vtk.vtkSphereSource()
#       source.SetCenter(0, 0, 0)
#       source.SetRadius(5.0)
#
#       # Create a mapper
#       mapper = vtk.vtkPolyDataMapper()
#       mapper.SetInputConnection(source.GetOutputPort())
#
#       # Create an actor
#       actor = vtk.vtkActor()
#       actor.SetMapper(mapper)
#
#       self.ren.AddActor(actor)
#
#       self.ren.ResetCamera()
#
#       self.frame.setLayout(self.vl)
#       self.setCentralWidget(self.frame)
#
#       self.show()
#       self.iren.Initialize()
#
#
# if __name__ == "__main__":
#
#   app = QApplication(sys.argv)
#
#   window = MainWindow()
#
#   sys.exit(app.exec_())




	# def add_cube(self, renderer, x_len, y_len, z_len, x_pos=0.0 , y_pos=0.0, z_pos=0.0, color=[0.0, 0.0, 1.0], opacity=1.0, write_stl=0, file_name="custom_stl"):
	# 	global actors
	# 	global custom_dict
	#
	# 	# create cube
	# 	cube = vtk.vtkCubeSource()
	# 	cube.SetXLength(x_len)
	# 	cube.SetYLength(y_len)
	# 	cube.SetZLength(z_len)
	# 	#
	#
	# 	# mapper
	# 	mapper = vtk.vtkPolyDataMapper()
	# 	mapper.SetInputConnection(cube.GetOutputPort())
	#
	# 	# actor
	# 	actor = vtk.vtkActor()
	# 	actor.SetMapper(mapper)
	# 	actor.SetPosition(x_pos, y_pos, z_pos)# stl is written at wrong position!
	# 	actor.GetProperty().SetColor(color)
	# 	actor.GetProperty().SetOpacity(opacity)
	#
	# 	renderer.AddActor(actor)
	# 	renderer.ResetCamera()
	# 	actors.append(actor) # not required to be deleted!
	#
	# 	# store values:
	# 	one_entry = {}
	# 	one_entry["Name"]  = self.getName_of_Actor(actor)
	# 	one_entry["Type"]  = "Cube"
	# 	one_entry["x_len"] = x_len
	# 	one_entry["y_len"] = y_len
	# 	one_entry["z_len"] = z_len
	# 	one_entry["x_pos"] = x_pos
	# 	one_entry["y_pos"] = y_pos
	# 	one_entry["z_pos"] = z_pos
	# 	# orientation fehlt noch! (TODO)
	# 	# visibility fehlt noch
	# 	one_entry["color_red"] = color[0]
	# 	one_entry["color_green"] = color[1]
	# 	one_entry["color_blue"] = color[2]
	# 	one_entry["opacity"] = opacity
	# 	custom_dict.append(one_entry)
	#
	# 	self.iren.GetRenderWindow().Render() # refresh window!
	# 	if write_stl:
	# 		 cube.SetCenter([x_pos, y_pos, z_pos])
	# 		 stlWriter = vtk.vtkSTLWriter()
	# 		 stlWriter.SetFileName(file_name)
	# 		 stlWriter.SetInputConnection(cube.GetOutputPort())
	# 		 stlWriter.Write()


		#self.iren.ReInitialize()
		#self.iren.GetRenderWindow().Render() # refresh window!

		# Load as json:
		#self.addDict_fromFile("MiniPick_json.txt")

		# # save as json:
		# self.dictToFile("MiniPick_json.txt")

		# remove and change:
		#self.setActorValues_sss(self.getName_of_Actor(self.ren.GetActors().GetLastActor()))





		# [dict, key] = self.getDictbyName("Robot", "kr6")
		# [theta_i, d_i, a_i, alpha_i] =  dict["dh_table"]
		# joints = [ dict["q0"]+self.deg2rad(self.ui.q1_slider.value()), dict["q1"]+self.deg2rad(self.ui.q2_slider.value()), dict["q2"], dict["q3"], dict["q4"], dict["q5"]]
		# res = self.calc_dh(joints, d_i, a_i, alpha_i)
		# T01 = self.calc_0Ti(res, i=0)
		# T02 = np.round(self.calc_0Ti(res, i=1), 1)
		#
		# rot2 = np.linalg.inv(T02[0:3,0:3]) #np.linalg.inv
		# l1 = { "name": "l1", "pos": T01[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T01[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# l2 = { "name": "l2", "pos":  T02[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": rot2, "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# self.addPose(self.ren, l1)
		# self.addPose(self.ren, l2)
		#
		# self.updateJoint(1, dict, T01[0:3,0:3],  T01[0:3, 3])
		# self.updateJoint(2, dict, rot2,  T02[0:3, 3])
		# self.update_list()

		# [dict, key] = self.getDictbyName("Robot", "kr6")
		# [theta_i, d_i, a_i, alpha_i] =  dict["dh_table"]
		# joints = [ dict["q0"]+self.deg2rad(self.ui.q1_slider.value()), dict["q1"]+self.deg2rad(self.ui.q2_slider.value()), dict["q2"], dict["q3"], dict["q4"], dict["q5"]]
		# res = self.calc_dh(joints, d_i, a_i, alpha_i)
		# T01 = self.calc_0Ti(res, i=0)
		# T02 = np.round(self.calc_0Ti(res, i=1), 1)
		#
		# rot2 = np.linalg.inv(T02[0:3,0:3])
		# l1 = { "name": "l1", "pos": T01[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": T01[0:3,0:3], "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# l2 = { "name": "l2", "pos":  T02[0:3, 3], "rot": [0.0, 0.0, 0.0], "rotation_3x3": rot2, "xlabel": "", "ylabel": "", "zlabel": "", "arrow_length": [150, 150, 150]}
		# self.addPose(self.ren, l1)
		# #self.addPose(self.ren, l2)
		#
		# self.updateJoint(1, dict, T01[0:3,0:3],  T01[0:3, 3])
		# self.updateJoint(2, dict, T02[0:3,0:3],  T02[0:3, 3])
		# self.update_list()
