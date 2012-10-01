// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef MAIN_H
#define MAIN_H

#ifdef __APPLE__
	#include <GL/glfw.h>
#else
	#include <GL/glew.h>
	#include <GL/glut.h>
#endif


//#include <stdio.h>

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "glslUtility.h"
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "image.h"
#include "raytraceKernel.h"
#include "utilities.h"
#include "scene.h"



using namespace std;

//INTERACTION
int currentX = 0;
int currentY = 0;
float rotateSpeed = 0.02;
float maxAngle = 40;
float minAngle = -40;
bool dragging = false;
bool rotating = false;
float dragspeed = 0.05;

//-------------------------------
//----------PATHTRACER-----------
//-------------------------------
int * effectiveRayMap = NULL;
ray * initialRayMap = NULL;
float * bMap = NULL; 
unsigned char * allmaps = NULL;
int mapsize = 0;

scene* renderScene;
camera* renderCam;
int targetFrame;
int iterations;
bool finishedRender;
bool singleFrameMode;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=800; int height=800;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
	void processSpecialKeys(int key, int x, int y); 
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	void mousewheel(int wheel, int direction, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
void initMaps();
void clearImage();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif