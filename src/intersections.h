// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>
#define ANGTORAD 0.01745329251994
#define boxEpsilon 0.0005
#define sphereEpsilon 0.0001
//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t - 0.0005)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors. 
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	//size = 1.0
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
	ray rt; rt.origin = ro; rt.direction = rd;
	   

	int trace_min = -1, trace_max = -1; // 0 x-0.5 1 x0.5 2 y-0.5 3 y0.5 4 z-0.5 5 z0.5
	float min = -1000000, max = 10000000;
	float tmpmin, tmpmax;
	//x planes

	if(epsilonCheck(rd.x, 0))
	{
		if(ro.x < -0.5 || ro.x > 0.5)
			return -1;
	}
	else
	{
		float tmp1 = (-0.5 - ro.x) / (rd.x);
		float tmp2 = (0.5 - ro.x ) / (rd.x);

	

		float tmp_trace_min = 0;
		tmpmin = tmp1;
		if(tmpmin > tmp2)
		{
			tmp_trace_min = 1;
			tmpmin = tmp2;
		}

		float tmp_trace_max = 1;
		tmpmax = tmp2;
		if(tmpmax < tmp1)
		{
			tmp_trace_max = 0;
			tmpmax = tmp1;
		}


		if(tmpmax < float(boxEpsilon))
			return -1;
		//if(tmpmin > max || tmpmax < min)
			//return -1;

		if(min < tmpmin)
		{
			trace_min = tmp_trace_min;
			min = glm::max(tmpmin,0.0f);
		}

		if(max > tmpmax)
		{
			trace_max = tmp_trace_max;
			max = tmpmax;
		}

	}

	

	if(epsilonCheck(rd.y, 0))
	{
		if(ro.y < -0.5 || ro.y > 0.5)
			return -1;
	}
	else
	{

		float tmp1 = (-0.5 - ro.y) / (rd.y);
		float tmp2 = (0.5 - ro.y) / (rd.y);
		
		float tmp_trace_min = 2;
		tmpmin = tmp1;
		if(tmpmin > tmp2)
		{
			tmp_trace_min = 3;
			tmpmin = tmp2;
		}

		float tmp_trace_max = 3;
		tmpmax = tmp2;
		if(tmpmax < tmp1)
		{
			tmp_trace_max = 2;
			tmpmax = tmp1;
		}

		if(tmpmax < float(boxEpsilon))
			return -1;

		if(tmpmin > max || tmpmax < min)
			return -1;

		if(min < tmpmin)
		{
			trace_min = tmp_trace_min;
			min = glm::max(tmpmin,0.0f);
		}

		if(max > tmpmax)
		{
			trace_max = tmp_trace_max;
			max = tmpmax;
		}
		
	}


	if(epsilonCheck(rd.z,0))
	{
		if(ro.z < -0.5 || ro.z > 0.5)
			return -1;
	}
	else
	{

		float tmp1 = (-0.5 - ro.z) / (rd.z);
		float tmp2 = (0.5 - ro.z) / (rd.z);
		

		
		float tmp_trace_min = 4;
		tmpmin = tmp1;
		if(tmpmin > tmp2)
		{
			tmp_trace_min = 5;
			tmpmin = tmp2;
		}

		float tmp_trace_max = 5;
		tmpmax = tmp2;
		if(tmpmax < tmp1)
		{
			tmp_trace_max = 4;
			tmpmax = tmp1;
		}

		if(tmpmax < float(boxEpsilon))
			return -1;

		if(tmpmin > max || tmpmax < min)
			return -1;

		if(min < tmpmin)
		{
			trace_min = tmp_trace_min;
			min = glm::max(tmpmin,0.0f);
		}

		if(max > tmpmax)
		{
			trace_max = tmp_trace_max;
			max = tmpmax;
		}

	}


	int t;
	if(min >= 0)
	{
		intersectionPoint = getPointOnRay(rt, min);
		t = trace_min;
		
	}
	else
	{
		t = trace_max;
		intersectionPoint = getPointOnRay(rt, max);
		
	}

		switch(t)
		{
		case 0: normal = glm::vec3(-1,0,0);
			break;
		case 1: normal = glm::vec3(1,0,0);
			break;
		case 2: normal = glm::vec3(0,-1,0);
			break;
		case 3: normal = glm::vec3(0,1,0);
			break;
		case 4: normal = glm::vec3(0,0,-1);
			break;
		case 5: normal = glm::vec3(0,0,1);
			break;
		}


	//value[index] = normal;

	
	if(min < 0)
		normal *= -1;

	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(intersectionPoint, 1.0));
	intersectionPoint = realIntersectionPoint;
	normal = multiplyMV(box.transform, glm::vec4(normal,0.0));
	normal = glm::normalize(normal);

	return glm::length(r.origin - realIntersectionPoint);


    
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < float(sphereEpsilon)){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }     

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);   
        
  return glm::length(r.origin - realIntersectionPoint);
}


__host__ __device__ glm::vec2 sphereTextureMap(staticGeom& sphere, glm::vec3& intersectionPoint)
{
	
	glm::vec3 localintersect = multiplyMV(sphere.inverseTransform, glm::vec4(intersectionPoint,1.0f));
	glm::vec2 uv;
	glm::vec3 N = glm::normalize(localintersect);
	
	glm::vec3 up = glm::vec3(0, 1.0 ,0);
	glm::vec3 forward = glm::vec3(0,0,1);

	float phi = max(acos(glm::dot(up,N)),0.0f);
	uv.y = phi / PI;

	float tmp = glm::dot(N,forward) / sin(phi);
	tmp = tmp > 1.0 ? 1.0 : tmp;
	tmp = tmp < -1.0 ? -1.0 : tmp;

	float theta = acos(tmp) / (2.0 * PI);
	theta = theta > 1.0 ? 1.0 : theta;

	if(glm::dot(glm::cross(up, forward), N) > 0.0)
	    uv.x= theta;
	else
		uv.x= 1.0 -theta;

	return uv;
	
	/*glm::vec3 localintersect = multiplyMV(sphere.inverseTransform, glm::vec4(intersectionPoint,1.0f));
	glm::vec2 uv;
	glm::vec4 N = glm::vec4(localintersect, 0);
	glm::vec4 up = glm::vec4(0, 1.0 ,0,0);
	glm::vec4 forward = glm::vec4(0,0,1,0);

	float phi = acos(glm::dot(up,N));
	uv.y = phi / PI;

	float tmp = glm::dot(N,forward) / sin(phi);
	tmp = tmp > 1.0 ? 1.0 : tmp;
	tmp = tmp < -1.0 ? -1.0 : tmp;

	float theta = acos(tmp) / (2.0 * PI);
	theta = theta > 1.0 ? 1.0 : theta;

	if(glm::dot(glm::cross(glm::vec3(up),glm::vec3(forward)), glm::vec3(N)) > 0.0)
	    uv.x= theta;
	else
		uv.x= 1.0 -theta;

	return uv;*/
}

__host__ __device__ glm::vec2 boxTextureMap(staticGeom& box, glm::vec3 & intersectionPoint)
{
	glm::vec3 localintersect = multiplyMV(box.inverseTransform, glm::vec4(intersectionPoint,1.0f));
	glm::vec2 uv;

	if(fabs(localintersect.x + 0.5f) < 0.01 || fabs(localintersect.x - .5f) < 0.01)
	{
	uv.x = glm::max(localintersect.y + 0.5f, 0.0f);
	uv.x = glm::min(uv.x,0.99999f);
	uv.y = glm::max(localintersect.z + 0.5f, 0.0f);
	uv.y = glm::min(uv.y,0.99999f);
	}

	if(fabs(localintersect.y + 0.5f) < 0.01 || fabs(localintersect.y - .5f) < 0.01)
	{
	uv.x = glm::max(localintersect.x + 0.5f, 0.0f);
	uv.x = glm::min(uv.x,0.99999f);
	uv.y = glm::max(localintersect.z + 0.5f, 0.0f);
	uv.y = glm::min(uv.y,0.99999f);
	}

	if(fabs(localintersect.z + 0.5f) < 0.01 || fabs(localintersect.z - .5f) < 0.01)
	{
	uv.x = glm::max(localintersect.x + 0.5f, 0.0f);
	uv.x = glm::min(uv.x,0.99999f);
	uv.y = glm::max(0.5f -localintersect.y , 0.0f);
	uv.y = glm::min(uv.y,0.99999f);

	}


	return uv;
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust. 
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f;   //x-y face
    float side2 = radii.z * radii.y * 4.0f;   //y-z face
    float side3 = radii.x * radii.z* 4.0f;   //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){


  
  return glm::vec3(0,0,0);
}

#endif