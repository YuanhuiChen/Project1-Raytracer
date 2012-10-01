// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__global__  void raycastFromCameraKernel(cameraData cam, ray * rayMap, int * effectiveMap, float * bMap){


	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	ray r;


	glm::vec3 horizontal = glm::normalize(glm::cross(cam.view,cam.up));

	// arrayv[index] = horizontal;

	glm::vec3 vertical =  -1.0f * glm::normalize(glm::cross(horizontal, cam.view));
	//arrayv[index] = vertical;

	float depth = 0.5 * cam.resolution.y / tan( ANGTORAD * cam.fov.y);
	float depthX = 0.5 * cam.resolution.x / tan( ANGTORAD * cam.fov.x);

	if(depthX > depth)
		depth = depthX;

	glm::vec3 hitPoint = cam.position + depth * cam.view + (x - 0.5f * cam.resolution.x) * horizontal + (y - 0.5f * cam.resolution.y) * vertical;
	r.origin = cam.position;
	r.direction = glm::normalize((hitPoint - cam.position));
	r.inside = false;
	rayMap[index] = r;
	effectiveMap[index] = 1;
	bMap[index] = 1;

}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		image[index] = glm::vec3(0,0,0);
	}
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image,int iterations){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	float inverseIterations = 1.0f / float(iterations); 
	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;      
		color.x = image[index].x*255.0 * inverseIterations;
		color.y = image[index].y*255.0 * inverseIterations;
		color.z = image[index].z*255.0 * inverseIterations;

		//color.x = image[index].x*255.0;
		//color.y = image[index].y*255.0;
		//color.z = image[index].z*255.0;

		if(color.x>255){
			color.x = 255;
		}

		if(color.y>255){
			color.y = 255;
		}

		if(color.z>255){
			color.z = 255;
		}

		// Each thread writes one pixel location in the texture (textel)
		PBOpos[index].w = 0;
		PBOpos[index].x = color.x;     
		PBOpos[index].y = color.y;
		PBOpos[index].z = color.z;
	}
}


__host__ __device__ bool testBlock(staticGeom * geoms, int lightNo, int numberOfGeoms, ray shootRay, float & attenuation, material *  mats)
{



	float distance = -1;
	int objid = -1;
	attenuation = 1.0;

	glm::vec3 intersect;
	glm::vec3 tmpPoint;
	glm::vec3 tmpNorm;

	bool hasblock = false;

	for(int i = 0; i < numberOfGeoms; i++)
	{

		float tmpDis = -1;
		switch(geoms[i].type)
		{
		case SPHERE:
			{

				tmpDis = sphereIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);

				break;
			}
		case CUBE:
			tmpDis = boxIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);
			// raysvalue[index] = shootRay.direction;
			break;

		}

		if(tmpDis != -1 )
		{

			if((tmpDis < distance || distance == -1 ))
			{
				distance = tmpDis;
				objid = i;
				intersect = tmpPoint;

			if(mats[geoms[i].materialid].hasRefractive)
				attenuation *= 0.9;

			}
		}
	}

	//colors[index].x = objid;
	//colors[index].y = lightNo;
	///colors[index].z = distance;
	if(objid != lightNo && objid != -1)
	{
		if(!(mats[geoms[objid].materialid].hasRefractive))
			attenuation = 0.0;
		return true;
	}
	else
		return false;


	/*attenuation = 1.0;

	glm::vec3 intersect;
	glm::vec3 tmpPoint;
	glm::vec3 tmpNorm;

	float distance = -1;
	float light_dis = glm::length(geoms[lightNo].translation - shootRay.origin);
	for(int i = 0; i < numberOfGeoms; i++)
	{

		float tmpDis = -1;
		switch(geoms[i].type)
		{
		case SPHERE:
			{

				tmpDis = sphereIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);

				break;
			}
		case CUBE:
			tmpDis = boxIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);
			// raysvalue[index] = shootRay.direction;
			break;

		}

		if(tmpDis != -1 )
		{

			if(i != lightNo && tmpDis + 0.007f < light_dis && !mats[geoms[i].materialid].hasRefractive)
			{
				attenuation = 0;
				return true;
			}
			if(mats[i].hasRefractive)
				attenuation *= 0.9;

		}
	}

	return false;*/

}

__device__ bool ChooseReflect(glm::vec2 resolution, float time, int x, int y)
{
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);

	if(u01(rng) > 0.5)
		return true;
	else
		return false;
}
/*__device__ float ChooseReflect(glm::vec2 resolution, float time, int x, int y)
{
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(-1,1);
	return u01(rng);

}*/
__device__ glm::vec3 ReadMapTexture(int mapid,staticGeom & intersectObj,glm::vec3 & intersectPoint, Map * maps, unsigned char * mapdata)
{
	glm::vec3 value;

	switch(intersectObj.type)
	{
	case SPHERE:
		{
			int shift = 0;
			for(int i = 0; i < mapid; i++)
				shift += maps[i].depth * maps[i].height * maps[i].width;
			Map map = maps[mapid];
			glm::vec2 uv = sphereTextureMap(intersectObj, intersectPoint);
			int xx = (int)(uv.x * map.width);
			int yy = (int)(uv.y * map.height);
			int index = shift + (xx + yy * map.width) * map.depth;
			//intersectMaterial.color = glm::vec3(float(map.mapptr[colorIndex]) / 255.0,(float)(map.mapptr[colorIndex + 1]) / 255.0,(float)map.mapptr[colorIndex + 2] / 255.0);
			value.x = (float)(mapdata[index]) / 255.0;
			value.y = (float)(mapdata[index + 1]) / 255.0;
			value.z = (float)mapdata[index + 2] / 255.0;

			break;
		}
	case CUBE:
		{
			int shift = 0;
			for(int i = 0; i < mapid; i++)
				shift += maps[i].depth * maps[i].height * maps[i].width;
			Map map = maps[mapid];
			glm::vec2 uv = boxTextureMap(intersectObj, intersectPoint);
			int xx = (int)(uv.x * map.width);
			int yy = (int)(uv.y * map.height);
			int index = shift + (xx + yy * map.width) * map.depth;
			//intersectMaterial.color = glm::vec3(float(map.mapptr[colorIndex]) / 255.0,(float)(map.mapptr[colorIndex + 1]) / 255.0,(float)map.mapptr[colorIndex + 2] / 255.0);
			value.x = (float)(mapdata[index]) / 255.0;
			value.y = (float)(mapdata[index + 1]) / 255.0;
			value.z = (float)mapdata[index + 2] / 255.0;

		}

	}

	return value;
}
__device__ glm::vec3 ReadMapNormal(int mapid,staticGeom & intersectObj,glm::vec3 & intersectPoint, Map * maps, unsigned char * mapdata)
{
	glm::vec3 value;

	switch(intersectObj.type)
	{
	case SPHERE:
		{
			int shift = 0;
			for(int i = 0; i < mapid; i++)
				shift += maps[i].depth * maps[i].height * maps[i].width;
			Map map = maps[mapid];
			glm::vec2 uv = sphereTextureMap(intersectObj, intersectPoint);
			int xx = (int)(uv.x * map.width);
			int yy = (int)(uv.y * map.height);
			int center = shift + (xx + yy * map.width) * map.depth;
			int left = glm::max(center - map.depth, 0);
			int top = glm::max(center - map.width * map.depth, 0);

			value.x = (float)(mapdata[center] - mapdata[top]) / 255.0;
			value.y = (float)(mapdata[center] - mapdata[left]) / 255.0;
			value.z = 0.2;
			break;
		}
	case CUBE:
		{
			int shift = 0;
			for(int i = 0; i < mapid; i++)
				shift += maps[i].depth * maps[i].height * maps[i].width;
			Map map = maps[mapid];
			glm::vec2 uv = boxTextureMap(intersectObj, intersectPoint);
			int xx = (int)(uv.x * map.width);
			int yy = (int)(uv.y * map.height);
			int center  = shift + (xx + yy * map.width) * map.depth;
			int left = glm::clamp(center - map.depth, 0, shift + (map.width*map.height-1) * map.depth);
			int top = glm::clamp(center - map.width * map.depth, 0, shift + (map.width*map.height-1) * map.depth);

			value.x = (float)(mapdata[center] - mapdata[top]) / 255.0;
			value.y = (float)(mapdata[center] - mapdata[left]) / 255.0;
			value.z = 0.2;

			break;
		}

	}

	return value;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(ray * rayMap, glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
	staticGeom* geoms, int numberOfGeoms, material * materials, Map * cudatextmap, unsigned char * cudatextmapdata, float * bMap, int * effectMap,int seed, int currentDepth){

		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;


		int index = x + (y * resolution.x);

		//int tmpindex = index;
		//initiallize bMatrix


		if(effectMap[index] == 0)
			return;
		ray shootRay = rayMap[index];
		glm::vec3 intersectPoint(0,0,0);
		glm::vec3 intersectNorm(0,0,0);

		int objid;

		float distance = -1;
		for(int i = 0; i < numberOfGeoms; i++)
		{
			float tmpDis = -1;
			glm::vec3 tmpPoint;
			glm::vec3 tmpNorm;
			switch(geoms[i].type)
			{
			case SPHERE:
				{

					tmpDis = sphereIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);

				}
				break;
			case CUBE:
				tmpDis = boxIntersectionTest(geoms[i],shootRay,tmpPoint, tmpNorm);
				break;

			}

			if(tmpDis != -1 && (tmpDis < distance || distance == -1 ))
			{
				distance = tmpDis;
				intersectPoint = tmpPoint;
				intersectNorm = tmpNorm;
				objid = i;
			}
		}


		//if have no intersection
		if(distance == -1)
		{
			int parelleltolight = -1;
			for(int i = 0; i < numberOfGeoms; i++)
			{

				if(materials[geoms[i].materialid].emittance > 0)
				{
					glm::vec3 lightdir = glm::normalize(geoms[i].translation - shootRay.origin);
					if(glm::dot(lightdir,shootRay.direction) > 0.99)
					{
						parelleltolight = i;
					}
				}
			}

			if(parelleltolight != -1)
			{
				colors[index] += materials[geoms[parelleltolight].materialid].color * bMap[index];
				effectMap[index] = 0;
			}

			return;
		}


		material intersectMaterial;
		staticGeom intersectObj = geoms[objid];
		intersectMaterial = materials[intersectObj.materialid];

		if(intersectObj.texturemapid != -1)
		intersectMaterial.color = ReadMapTexture(intersectObj.texturemapid,intersectObj, intersectPoint, cudatextmap, cudatextmapdata);
		if(intersectObj.normalmapid != -1)
		{
	     intersectNorm += 2.0f * glm::normalize(ReadMapNormal(intersectObj.normalmapid,intersectObj, intersectPoint, cudatextmap, cudatextmapdata));
		//intersectNorm += ReadMapNormal(intersectObj.normalmapid,intersectObj, intersectPoint, cudatextmap, cudatextmapdata);
		intersectNorm = glm::normalize(intersectNorm);
		//intersectNorm = glm::normalize(ReadMapNormal(intersectObj.normalmapid,intersectObj, intersectPoint, cudatextmap, cudatextmapdata));
		}
		float b = bMap[index];

		//colors[index].y = 1;

		//if Hit Light
		if(intersectMaterial.emittance  > 0)
		{
			colors[index] += b * intersectMaterial.color;
			effectMap[index] = 0;
			return;
		}

		//normal condition
		ray lightray;
		glm::vec3 outcolor = glm::vec3(0,0,0);

for(int i = 0; i < numberOfGeoms; i++)
		{
			//test block
			material lightMaterial = materials[geoms[i].materialid];
			if(lightMaterial.emittance > 0)
			{


				glm::vec3 tmpPoint, tmpNorm;
				lightray.direction = glm::normalize(geoms[i].translation - intersectPoint);
				lightray.origin = intersectPoint;
				float attenuation = 1.0;
				//if block or light ray cannot reach
				testBlock(geoms, i, numberOfGeoms,lightray,attenuation,materials);

				
				float tmpdotvalue = max(glm::dot(intersectNorm, lightray.direction),0.0001);
				//lambert diffuse
				outcolor +=  attenuation * lightMaterial.color * ( intersectMaterial.color * tmpdotvalue);

				//specular
				if(intersectMaterial.hasReflective)
				{
				tmpdotvalue = max(glm::dot( calculateReflectionDirection(intersectNorm,-1.0f * lightray.direction), -1.0f * shootRay.direction), 0.0);
				outcolor += attenuation*glm::pow(tmpdotvalue,intersectMaterial.specularExponent) * lightMaterial.color * intersectMaterial.color;
				}

			}
		}



		colors[index] += b * outcolor;


		//decide next ray

		ray reflectRay, refractRay;

		Fresnel fresnel;


		if(intersectMaterial.hasRefractive || intersectMaterial.hasReflective)
		{


			effectMap[index] = 1;

			if(intersectMaterial.hasRefractive)
			{
				if(shootRay.inside)
				{
					reflectRay.direction = calculateReflectionDirection(-1.0f * intersectNorm,shootRay.direction);
					fresnel = calculateFresnel(-1.0f * intersectNorm, shootRay.direction, intersectMaterial.indexOfRefraction, refractRay.direction);
				}
				else
				{

					reflectRay.direction = calculateReflectionDirection(intersectNorm,shootRay.direction);
					fresnel = calculateFresnel(intersectNorm, shootRay.direction, 1.0f / intersectMaterial.indexOfRefraction, refractRay.direction);
				}


				refractRay.origin = intersectPoint  + 0.005f * refractRay.direction;
				refractRay.inside = !shootRay.inside;
				reflectRay.inside = shootRay.inside;
				reflectRay.origin = intersectPoint;



				if(fresnel.reflectionCoefficient == 1 )
				{
					b *= fresnel.reflectionCoefficient;
					rayMap[index] = reflectRay;
					//	colors[index] = glm::vec3(1.0, 0.0, 0.0); 
					//effectMap[index] = 0;
				}
				else
				{

					b *= fresnel.transmissionCoefficient;
					rayMap[index] = refractRay;
				}
			}
			else
			{
				reflectRay.direction = calculateReflectionDirection(intersectNorm,shootRay.direction);
				reflectRay.origin = intersectPoint;
				reflectRay.inside = shootRay.inside;
				b *= 0.3;
				rayMap[index] = reflectRay;

			}

		}
		else
		{

			effectMap[index] = 0;
		}
		//colors[index] = time * generateRandomNumberFromThread(resolution, time, x, y);
		bMap[index] = b;
		
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,Map * maps, int numberOfMaps){

	int traceDepth = 5; //determines how many bounces the raytracer traces
	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		newStaticGeom.normalmapid = geoms[i].noramlmapId;
		newStaticGeom.texturemapid = geoms[i].texturemapId;
		geomList[i] = newStaticGeom;
	}

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	material * cudamaters = NULL;
	cudaMalloc((void **) & cudamaters,numberOfMaterials * sizeof(material));
	cudaMemcpy(cudamaters, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);



	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	//package maps

	//package bMap

	float * cudabMap = NULL;
	cudaMalloc((void **) & cudabMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float));
	cudaMemcpy(cudabMap, bMap,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float), cudaMemcpyHostToDevice);

	//package rayMap
	ray * cudaRayMap;
	cudaMalloc((void **) & cudaRayMap,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray));
	cudaMemcpy(cudaRayMap, initialRayMap,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray), cudaMemcpyHostToDevice);


	//package effectiveRayMap
	int * cudaEffective;
	cudaMalloc((void **) & cudaEffective,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(int));
	cudaMemcpy(cudaEffective,effectiveRayMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(int), cudaMemcpyHostToDevice);
	//kernel launches

	//package mapsdata
	unsigned char * cudatextmapdata = NULL;
	if(mapsize != 0)
	{
		cudaMalloc((void **) & cudatextmapdata, mapsize * sizeof(unsigned char));
		cudaMemcpy(cudatextmapdata, allmaps, mapsize * sizeof(unsigned char),cudaMemcpyHostToDevice);
	}
	//package maps
	Map * cudatextmap = NULL;
	if(numberOfMaps != 0)
	{
		cudaMalloc((void **) & cudatextmap, numberOfMaps * sizeof(Map));
		cudaMemcpy(cudatextmap, maps,numberOfMaps * sizeof(Map) , cudaMemcpyHostToDevice);
	}

	for(int i = 0; i <= traceDepth; i++)
	{
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaRayMap, renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaters, cudatextmap, cudatextmapdata,cudabMap,cudaEffective,time(NULL), i);
	}

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree(cudamaters);
	cudaFree(cudabMap);
	cudaFree(cudaRayMap);
	cudaFree(cudaEffective);
	cudaFree(cudatextmapdata);
	cudaFree(cudatextmap);
	delete geomList;

	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}

void generateRayMap(camera* renderCam,int frame)
{

	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;



	ray * cudaRayMap;
	cudaMalloc((void **) & cudaRayMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray));
	float * cudabMap;
	cudaMalloc((void **) & cudabMap,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float));
	int * cudaeffectMap;
	cudaMalloc((void **) & cudaeffectMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(int));
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, cudaRayMap, cudaeffectMap, cudabMap);

	cudaMemcpy(initialRayMap, cudaRayMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray), cudaMemcpyDeviceToHost);
	cudaMemcpy(effectiveRayMap,cudaeffectMap, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(bMap, cudabMap,(int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaFree(cudaRayMap);
	cudaFree(cudabMap);
	cudaFree(cudaeffectMap);
}

