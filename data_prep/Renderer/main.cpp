"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


#include <GL/glut.h>
#include <GL/glext.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <glm/mat4x4.hpp>
#include <png.h>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <math.h>

using std::stringstream;
using std::cout;
using std::endl;
using std::ends;
using std::vector;
using std::string;
using std::ifstream;

DEFINE_string(scene_id, "", "scene id");
DEFINE_string(frames_id, "", "frames id");
DEFINE_string(root_folder, "/mnt/vision/ScanNet/", "root folder");
DEFINE_int32(frame_stride, 1, "frame stride");
DEFINE_string(filename, "planes", "ply filename");

void displayCB();
void reshapeCB(int w, int h);
void timerCB(int millisec);
void keyboardCB(unsigned char key, int x, int y);
void mouseCB(int button, int stat, int x, int y);
void mouseMotionCB(int x, int y);

void initGL();
int  initGLUT(int argc, char **argv);
bool initSharedMem();
void clearSharedMem();
void initLights();
void toPerspective();
void draw();
void drawTest(vector<GLfloat> &vertices, vector<GLfloat> &colors, vector<GLuint> &indices);

const int   SCREEN_WIDTH    = 640;
const int   SCREEN_HEIGHT   = 480;
const float CAMERA_DISTANCE = 10.0f;
const int   TEXT_WIDTH      = 8;
const int   TEXT_HEIGHT     = 13;

int screenWidth;
int screenHeight;
double focalLength;
int numFrames;
double depthShift;

bool mouseLeftDown;
bool mouseRightDown;
bool mouseMiddleDown;
float mouseX, mouseY;
float cameraAngleX;
float cameraAngleY;
float cameraDistance;
float cameraX;
float cameraY;
float cameraZ;
int drawMode;
int maxVertices;
int maxIndices;
string scene_id;

int frame_index;

#ifdef _WIN32
PFNGLDRAWRANGEELEMENTSPROC pglDrawRangeElements = 0;
#define glDrawRangeElements pglDrawRangeElements
#endif



// cube ///////////////////////////////////////////////////////////////////////
//    v6----- v5
//   /|      /|
//  v1------v0|
//  | |     | |
//  | |v7---|-|v4
//  |/      |/
//  v2------v3

// vertex coords array for glDrawArrays() =====================================
// A cube has 6 sides and each side has 2 triangles, therefore, a cube consists
// of 36 vertices (6 sides * 2 tris * 3 vertices = 36 vertices). And, each
// vertex is 3 components (x,y,z) of floats, therefore, the size of vertex
// array is 108 floats (36 * 3 = 108).
GLfloat vertices1[] = { 1, 1, 1,  -1, 1, 1,  -1,-1, 1,      // v0-v1-v2 (front)
                       -1,-1, 1,   1,-1, 1,   1, 1, 1,      // v2-v3-v0

                        1, 1, 1,   1,-1, 1,   1,-1,-1,      // v0-v3-v4 (right)
                        1,-1,-1,   1, 1,-1,   1, 1, 1,      // v4-v5-v0

                        1, 1, 1,   1, 1,-1,  -1, 1,-1,      // v0-v5-v6 (top)
                       -1, 1,-1,  -1, 1, 1,   1, 1, 1,      // v6-v1-v0

                       -1, 1, 1,  -1, 1,-1,  -1,-1,-1,      // v1-v6-v7 (left)
                       -1,-1,-1,  -1,-1, 1,  -1, 1, 1,      // v7-v2-v1

                       -1,-1,-1,   1,-1,-1,   1,-1, 1,      // v7-v4-v3 (bottom)
                        1,-1, 1,  -1,-1, 1,  -1,-1,-1,      // v3-v2-v7

                        1,-1,-1,  -1,-1,-1,  -1, 1,-1,      // v4-v7-v6 (back)
                       -1, 1,-1,   1, 1,-1,   1,-1,-1 };    // v6-v5-v4

// normal array
GLfloat normals1[]  = { 0, 0, 1,   0, 0, 1,   0, 0, 1,      // v0-v1-v2 (front)
                        0, 0, 1,   0, 0, 1,   0, 0, 1,      // v2-v3-v0

                        1, 0, 0,   1, 0, 0,   1, 0, 0,      // v0-v3-v4 (right)
                        1, 0, 0,   1, 0, 0,   1, 0, 0,      // v4-v5-v0

                        0, 1, 0,   0, 1, 0,   0, 1, 0,      // v0-v5-v6 (top)
                        0, 1, 0,   0, 1, 0,   0, 1, 0,      // v6-v1-v0

                       -1, 0, 0,  -1, 0, 0,  -1, 0, 0,      // v1-v6-v7 (left)
                       -1, 0, 0,  -1, 0, 0,  -1, 0, 0,      // v7-v2-v1

                        0,-1, 0,   0,-1, 0,   0,-1, 0,      // v7-v4-v3 (bottom)
                        0,-1, 0,   0,-1, 0,   0,-1, 0,      // v3-v2-v7

                        0, 0,-1,   0, 0,-1,   0, 0,-1,      // v4-v7-v6 (back)
                        0, 0,-1,   0, 0,-1,   0, 0,-1 };    // v6-v5-v4

// color array
GLfloat colors1[]   = { 1, 1, 1,   1, 1, 0,   1, 0, 0,      // v0-v1-v2 (front)
                        1, 0, 0,   1, 0, 1,   1, 1, 1,      // v2-v3-v0

                        1, 1, 1,   1, 0, 1,   0, 0, 1,      // v0-v3-v4 (right)
                        0, 0, 1,   0, 1, 1,   1, 1, 1,      // v4-v5-v0

                        1, 1, 1,   0, 1, 1,   0, 1, 0,      // v0-v5-v6 (top)
                        0, 1, 0,   1, 1, 0,   1, 1, 1,      // v6-v1-v0

                        1, 1, 0,   0, 1, 0,   0, 0, 0,      // v1-v6-v7 (left)
                        0, 0, 0,   1, 0, 0,   1, 1, 0,      // v7-v2-v1

                        0, 0, 0,   0, 0, 1,   1, 0, 1,      // v7-v4-v3 (bottom)
                        1, 0, 1,   1, 0, 0,   0, 0, 0,      // v3-v2-v7

                        0, 0, 1,   0, 0, 0,   0, 1, 0,      // v4-v7-v6 (back)
                        0, 1, 0,   0, 1, 1,   0, 0, 1 };    // v6-v5-v4



// vertex array for glDrawElements() and glDrawRangeElement() =================
// Notice that the sizes of these arrays become samller than the arrays for
// glDrawArrays() because glDrawElements() uses an additional index array to
// choose designated vertices with the indices. The size of vertex array is now
// 24 instead of 36, but the index array size is 36, same as the number of
// vertices required to draw a cube.
GLfloat vertices2[] = { 1, 1, 1,  -1, 1, 1,  -1,-1, 1,   1,-1, 1,   // v0,v1,v2,v3 (front)
                        1, 1, 1,   1,-1, 1,   1,-1,-1,   1, 1,-1,   // v0,v3,v4,v5 (right)
                        1, 1, 1,   1, 1,-1,  -1, 1,-1,  -1, 1, 1,   // v0,v5,v6,v1 (top)
                       -1, 1, 1,  -1, 1,-1,  -1,-1,-1,  -1,-1, 1,   // v1,v6,v7,v2 (left)
                       -1,-1,-1,   1,-1,-1,   1,-1, 1,  -1,-1, 1,   // v7,v4,v3,v2 (bottom)
                        1,-1,-1,  -1,-1,-1,  -1, 1,-1,   1, 1,-1 }; // v4,v7,v6,v5 (back)

// normal array
GLfloat normals2[]  = { 0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,   // v0,v1,v2,v3 (front)
                        1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,   // v0,v3,v4,v5 (right)
                        0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,   // v0,v5,v6,v1 (top)
                       -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,   // v1,v6,v7,v2 (left)
                        0,-1, 0,   0,-1, 0,   0,-1, 0,   0,-1, 0,   // v7,v4,v3,v2 (bottom)
                        0, 0,-1,   0, 0,-1,   0, 0,-1,   0, 0,-1 }; // v4,v7,v6,v5 (back)

// color array
GLfloat colors2[]   = { 1, 1, 1,   1, 1, 0,   1, 0, 0,   1, 0, 1,   // v0,v1,v2,v3 (front)
                        1, 1, 1,   1, 0, 1,   0, 0, 1,   0, 1, 1,   // v0,v3,v4,v5 (right)
                        1, 1, 1,   0, 1, 1,   0, 1, 0,   1, 1, 0,   // v0,v5,v6,v1 (top)
                        1, 1, 0,   0, 1, 0,   0, 0, 0,   1, 0, 0,   // v1,v6,v7,v2 (left)
                        0, 0, 0,   0, 0, 1,   1, 0, 1,   1, 0, 0,   // v7,v4,v3,v2 (bottom)
                        0, 0, 1,   0, 0, 0,   0, 1, 0,   0, 1, 1 }; // v4,v7,v6,v5 (back)

// index array of vertex array for glDrawElements() & glDrawRangeElement()
GLubyte indices[]  = { 0, 1, 2,   2, 3, 0,      // front
                       4, 5, 6,   6, 7, 4,      // right
                       8, 9,10,  10,11, 8,      // top
                      12,13,14,  14,15,12,      // left
                      16,17,18,  18,19,16,      // bottom
                      20,21,22,  22,23,20 };    // back



// interleaved vertex array for glDrawElements() & glDrawRangeElements() ======
// All vertex attributes (position, normal, color) are packed together as a
// struct or set, for example, ((V,N,C), (V,N,C), (V,N,C),...).
// It is called an array of struct, and provides better memory locality.
GLfloat vertices3[] = { 1, 1, 1,   0, 0, 1,   1, 1, 1,              // v0 (front)
                       -1, 1, 1,   0, 0, 1,   1, 1, 0,              // v1
                       -1,-1, 1,   0, 0, 1,   1, 0, 0,              // v2
                        1,-1, 1,   0, 0, 1,   1, 0, 1,              // v3

                        1, 1, 1,   1, 0, 0,   1, 1, 1,              // v0 (right)
                        1,-1, 1,   1, 0, 0,   1, 0, 1,              // v3
                        1,-1,-1,   1, 0, 0,   0, 0, 1,              // v4
                        1, 1,-1,   1, 0, 0,   0, 1, 1,              // v5

                        1, 1, 1,   0, 1, 0,   1, 1, 1,              // v0 (top)
                        1, 1,-1,   0, 1, 0,   0, 1, 1,              // v5
                       -1, 1,-1,   0, 1, 0,   0, 1, 0,              // v6
                       -1, 1, 1,   0, 1, 0,   1, 1, 0,              // v1

                       -1, 1, 1,  -1, 0, 0,   1, 1, 0,              // v1 (left)
                       -1, 1,-1,  -1, 0, 0,   0, 1, 0,              // v6
                       -1,-1,-1,  -1, 0, 0,   0, 0, 0,              // v7
                       -1,-1, 1,  -1, 0, 0,   1, 0, 0,              // v2

                       -1,-1,-1,   0,-1, 0,   0, 0, 0,              // v7 (bottom)
                        1,-1,-1,   0,-1, 0,   0, 0, 1,              // v4
                        1,-1, 1,   0,-1, 0,   1, 0, 1,              // v3
                       -1,-1, 1,   0,-1, 0,   1, 0, 0,              // v2

                        1,-1,-1,   0, 0,-1,   0, 0, 1,              // v4 (back)
                       -1,-1,-1,   0, 0,-1,   0, 0, 0,              // v7
                       -1, 1,-1,   0, 0,-1,   0, 1, 0,              // v6
                        1, 1,-1,   0, 0,-1,   0, 1, 1 };            // v5


GLfloat verticesTest[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0};
GLfloat colorsTest[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
GLubyte indicesTest[] = {0, 1, 2};


vector<GLfloat> verticesVec;
vector<GLfloat> colorsVec;
vector<GLuint> indicesVec;
glm::mat4 transformation;



void screenshot(const string filename, const unsigned int width, const unsigned int height)
{
  //cout << "screenshot" << endl;
  
  //glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
  //glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, 0);
  //vector<unsigned char> data = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

  glReadBuffer(GL_BACK);
  vector<unsigned char> data(width * height * 3);
  glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, &data[0]);
  cv::Mat image(height, width, CV_8UC3, &data[0]);
  cv::flip(image, image, 0);
  cv::imwrite(filename, image);
}

///////////////////////////////////////////////////////////////////////////////
// draw cube at bottom-left corner with glDrawElements
// The main advantage of glDrawElements() over glDrawArray() is that
// glDrawElements() allows hopping around the vertex array with the associated
// index values.
// In a cube, the number of vertex data in the vertex array can be reduced to
// 24 vertices for glDrawElements().
// Note that you need an additional array (index array) to store how to traverse
// the vertext data. For a cube, we need 36 entries in the index array.
///////////////////////////////////////////////////////////////////////////////
void draw()
{
    // enable and specify pointers to vertex arrays
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glNormalPointer(GL_FLOAT, 0, normals2);
    glColorPointer(3, GL_FLOAT, 0, colors2);
    glVertexPointer(3, GL_FLOAT, 0, vertices2);

    glPushMatrix();
    glTranslatef(-3, -3, 0);                // move to bottom-left corner

    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, indices);

    glPopMatrix();

    glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}

void drawTest(vector<GLfloat> &vertices, vector<GLfloat> &colors, vector<GLuint> &indices)
{
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColorPointer(3, GL_FLOAT, 0, &colors[0]);
    glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);

    glPushMatrix();
    //glTranslatef(0, 0, -cameraDistance);
    
    glMultMatrixf(&transformation[0][0]);
    
    //glMultTransposeMatrixf(&transformation[0][0]);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

    glPopMatrix();

    glDisableClientState(GL_VERTEX_ARRAY);  // disable vertex arrays
    glDisableClientState(GL_COLOR_ARRAY);
}

// void init()
// {
//   verticesVec.clear();
//     verticesVec.push_back(-1);
//     verticesVec.push_back(-1);
//     verticesVec.push_back(0);
//     verticesVec.push_back(1);
//     verticesVec.push_back(-1);
//     verticesVec.push_back(0);
//     verticesVec.push_back(1);
//     verticesVec.push_back(1);
//     verticesVec.push_back(0);

//     colorsVec.clear();
//     colorsVec.push_back(1);
//     colorsVec.push_back(0);
//     colorsVec.push_back(0);
//     colorsVec.push_back(0);
//     colorsVec.push_back(1);
//     colorsVec.push_back(0);
//     colorsVec.push_back(0);
//     colorsVec.push_back(0);
//     colorsVec.push_back(1);

//     indicesVec.clear();
//     indicesVec.push_back(0);
//     indicesVec.push_back(1);
//     indicesVec.push_back(2);
// }

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  frame_index = -1;
  cameraX = 0;
  cameraY = 0;
  cameraZ = 0;
  
  scene_id = FLAGS_scene_id;
  
  string line;
  stringstream ply_filename;
  ply_filename << FLAGS_root_folder << scene_id << "/annotation/" << FLAGS_filename << ".ply";
  ifstream ply_file(ply_filename.str());
  string dump;
  int num_vertices = 0;
  int num_faces = 0;
  float vertex;
  int color;
  int index;
  int max_index = 0;
  if (ply_file.is_open()) {
    int line_index = 0;
    while (getline(ply_file, line)) {
      stringstream ss(line);
      if (line_index == 2) {
	ss >> dump;
	ss >> dump;
	ss >> num_vertices;
      } else if (line_index == 9) {
	ss >> dump;
	ss >> dump;
	ss >> num_faces;
      } else if (line_index >= 12 && line_index < 12 + num_vertices) {
	for (int c = 0; c < 3; c++) {
	  ss >> vertex;
	  verticesVec.push_back(GLfloat(vertex));
	}
	for (int c = 0; c < 3; c++) {
	  ss >> color;
	  colorsVec.push_back(GLfloat(color) / 255);
	}
      } else if (line_index >= 12 + num_vertices) {
	ss >> dump;
	for (int c = 0; c < 3; c++) {
	  ss >> index;
	  if (index > max_index)
	    max_index = index;
	  indicesVec.push_back(GLuint(index));
	}	
      } else if (line_index >= 12 + num_vertices + num_faces) {
	break;
      }
      line_index++;
    }
    cout << num_vertices << " " << num_faces << " " << max_index << endl;
  }

  string key;
  double number;
  
  string infoFilename = FLAGS_root_folder + scene_id + "/" + scene_id + ".txt";

  ifstream scene_file(infoFilename);
  if (scene_file.is_open()) {
    while (getline(scene_file, line)) {
      stringstream ss(line);
      ss >> key >> dump >> number;
      if (key == "depthHeight")
  	screenHeight = number;
      if (key == "depthWidth")
  	screenWidth = number; 
      if (key == "fx_depth")
  	focalLength = number;
      if (key == "numDepthFrames")
  	numFrames = ceil(number / FLAGS_frame_stride);
    }
  }
  depthShift = 1000.0;
  
  cout << screenHeight << '\t' << screenWidth << '\t' << focalLength << '\t' << numFrames << '\t' << endl;

    // check max of elements vertices and elements indices that your video card supports
    // Use these values to determine the range of glDrawRangeElements()
    // The constants are defined in glext.h

    // init global vars
    initSharedMem();

    // init GLUT and GL
    initGLUT(argc, argv);
    initGL();
  
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &maxVertices);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &maxIndices);

#ifdef _WIN32
    // get function pointer to glDrawRangeElements
    glDrawRangeElements = (PFNGLDRAWRANGEELEMENTSPROC)wglGetProcAddress("glDrawRangeElements");
#endif
    
    // the last GLUT call (LOOP)
    // window will be shown and display callback is triggered by events
    // NOTE: this call never return main().
    glutMainLoop(); /* Start GLUT event-processing loop */

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
// initialize GLUT for windowing
///////////////////////////////////////////////////////////////////////////////
int initGLUT(int argc, char **argv)
{
    // GLUT stuff for windowing
    // initialization openGL window.
    // it is called before any other GLUT routine
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);   // display mode

    glutInitWindowSize(screenWidth, screenHeight);  // window size

    glutInitWindowPosition(100, 100);               // window location

    // finally, create a window with openGL context
    // Window will not displayed until glutMainLoop() is called
    // it returns a unique ID
    int handle = glutCreateWindow(argv[0]);     // param is the title of window

    // register GLUT callback functions
    glutDisplayFunc(displayCB);
    glutTimerFunc(33, timerCB, 33);             // redraw only every given millisec
    glutReshapeFunc(reshapeCB);
    glutKeyboardFunc(keyboardCB);
    glutMouseFunc(mouseCB);
    glutMotionFunc(mouseMotionCB);

    return handle;
}



///////////////////////////////////////////////////////////////////////////////
// initialize OpenGL
// disable unused features
///////////////////////////////////////////////////////////////////////////////
void initGL()
{
    glShadeModel(GL_SMOOTH);                    // shading mathod: GL_SMOOTH or GL_FLAT
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);      // 4-byte pixel alignment

    // enable /disable features
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    //glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    //glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);

     // track material ambient and diffuse from surface color, call it before glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glClearColor(0, 0, 0, 0);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    glDepthFunc(GL_LEQUAL);

    //initLights();

    //glGenBuffers(1, &pbo);
    //glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    //glBufferData(GL_PIXEL_PACK_BUFFER, SCREEN_WIDTH * SCREEN_HEIGHT * 3, NULL, GL_DYNAMIC_READ);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}



///////////////////////////////////////////////////////////////////////////////
// initialize global variables
///////////////////////////////////////////////////////////////////////////////
bool initSharedMem()
{
    mouseLeftDown = mouseRightDown = mouseMiddleDown = false;
    mouseX = mouseY = 0;

    cameraAngleX = cameraAngleY = 0.0f;
    cameraDistance = CAMERA_DISTANCE;
    
    drawMode = 0; // 0:fill, 1: wireframe, 2:points
    maxVertices = maxIndices = 0;

    return true;
}



///////////////////////////////////////////////////////////////////////////////
// clean up global vars
///////////////////////////////////////////////////////////////////////////////
void clearSharedMem()
{
  //glDeleteBuffers(1, &pbo);
}



///////////////////////////////////////////////////////////////////////////////
// initialize lights
///////////////////////////////////////////////////////////////////////////////
void initLights()
{
    // set up light colors (ambient, diffuse, specular)
    GLfloat lightKa[] = {.2f, .2f, .2f, 1.0f};  // ambient light
    GLfloat lightKd[] = {.7f, .7f, .7f, 1.0f};  // diffuse light
    GLfloat lightKs[] = {1, 1, 1, 1};           // specular light
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

    // position the light
    float lightPos[4] = {0, 0, 20, 1}; // positional light
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
}



// ///////////////////////////////////////////////////////////////////////////////
// // set camera position and lookat direction
// ///////////////////////////////////////////////////////////////////////////////
// void setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
// {
//     glMatrixMode(GL_MODELVIEW);
//     glLoadIdentity();
//     gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, 0, 1, 0); // eye(x,y,z), focal(x,y,z), up(x,y,z)
// }



///////////////////////////////////////////////////////////////////////////////
// set projection matrix as orthogonal
///////////////////////////////////////////////////////////////////////////////
void toOrtho()
{
    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)screenWidth, (GLsizei)screenHeight);

    // set orthographic viewing frustum
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, screenWidth, 0, screenHeight, -1, 1);

    // switch to modelview matrix in order to set scene
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}



///////////////////////////////////////////////////////////////////////////////
// set the projection matrix as perspective
///////////////////////////////////////////////////////////////////////////////
void toPerspective()
{
    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)screenWidth, (GLsizei)screenHeight);

    // set perspective viewing frustum
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float fov = 2 * atan(float(screenHeight) / 2 / focalLength) / 3.14 * 180;
    gluPerspective(fov, (float)(screenWidth)/screenHeight, 0.1f, 100.0f); // FOV, AspectRatio, NearClip, FarClip

    // switch to modelview matrix in order to set scene
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


//=============================================================================
// CALLBACKS
//=============================================================================

void displayCB()
{
    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // save the initial ModelView matrix before modifying ModelView matrix
    glPushMatrix();

    // tramsform camera
    glTranslatef(cameraX, cameraY, cameraZ);
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 1, 0);   // heading

    drawTest(verticesVec, colorsVec, indicesVec);

    glPopMatrix();

    glutSwapBuffers();

    if (frame_index < numFrames) {
      if (frame_index >= 0) {
	stringstream filename_ss;
	filename_ss << FLAGS_root_folder << scene_id << "/annotation/segmentation" + FLAGS_frames_id + "/" << frame_index << ".png";
	screenshot(filename_ss.str(), screenWidth, screenHeight);
      }
      frame_index += 1;

      if (frame_index >= numFrames)
	exit(0);

      string line;
      double number;
      stringstream filename_ss;
      filename_ss << FLAGS_root_folder << scene_id << "/frames" << FLAGS_frames_id << "/pose/" << frame_index << ".txt";
	
      ifstream pose_file(filename_ss.str());
      if (pose_file.is_open()) {
	for (int line_index = 0; line_index < 3; line_index++) {
	  getline(pose_file, line);
	  stringstream ss(line);
	  for (int c = 0; c < 3; c++) {
	    ss >> number;
	    transformation[c][line_index] = GLfloat(number);
	  }
	  ss >> number;
	  transformation[3][line_index] = GLfloat(number);
	}
      }

      for (int c = 0; c < 3; c++)
	transformation[c][3] = 0.0;
      transformation[3][3] = 1.0;

      transformation = glm::inverse(transformation);
  
      glm::mat4 coordinates;
      for (int c = 0; c < 4; c++)
	for (int d = 0; d < 4; d++)
	  coordinates[c][d] = 0;
      coordinates[0][0] = 1;
      coordinates[1][1] = -1;
      coordinates[2][2] = -1;
      coordinates[3][3] = 1;
      transformation = coordinates * transformation;
    }
}


void reshapeCB(int w, int h)
{
    screenWidth = w;
    screenHeight = h;
    toPerspective();
}


void timerCB(int millisec)
{
    glutTimerFunc(millisec, timerCB, millisec);
    glutPostRedisplay();
}


void keyboardCB(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27: // ESCAPE
        clearSharedMem();
        exit(0);
        break;

    case 'm': // switch rendering modes (fill -> wire -> point)
    case 'M':
        drawMode = ++drawMode % 3;
        if(drawMode == 0)        // fill mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }
        else if(drawMode == 1)  // wireframe mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }
        else                    // point mode
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }
        break;
    case 'w':
      cameraAngleX += 10;
      break;
    case 's':
      cameraAngleX -= 10;
      break;
    case 'a':
      cameraAngleY += 10;
      break;
    case 'd':
      cameraAngleY -= 10;
      break;
    case 'h':
      cameraX += 0.3;
      break;
    case 'l':
      cameraX -= 0.3;
      break;
    case 'j':
      cameraY += 0.3;
      break;
    case 'k':
      cameraY -= 0.3;
      break;      
    case 'u':
      cameraZ += 0.3;
      break;
    case 'i':
      cameraZ -= 0.3;
      break;      
    default:
      break;
      ;
    }
}


void mouseCB(int button, int state, int x, int y)
{
    mouseX = x;
    mouseY = y;

    if(button == GLUT_LEFT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseLeftDown = true;
        }
        else if(state == GLUT_UP)
            mouseLeftDown = false;
    }

    else if(button == GLUT_RIGHT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseRightDown = true;
        }
        else if(state == GLUT_UP)
            mouseRightDown = false;
    }

    else if(button == GLUT_MIDDLE_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseMiddleDown = true;
        }
        else if(state == GLUT_UP)
            mouseMiddleDown = false;
    }
}


void mouseMotionCB(int x, int y)
{
    if(mouseLeftDown)
    {
        cameraAngleY += (x - mouseX);
        cameraAngleX += (y - mouseY);
        mouseX = x;
        mouseY = y;
    }
    if(mouseRightDown)
    {
        cameraDistance -= (y - mouseY) * 0.2f;
        mouseY = y;
    }
}
