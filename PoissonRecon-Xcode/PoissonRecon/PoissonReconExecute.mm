//
//  PoissonReconExecute.mm
//  PoissonRecon
//
//  Created by Aaron Thompson on 5/8/18.
//  Copyright © 2018 Standard Cyborg. All rights reserved.
//

#import "PoissonReconExecute.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "MyTime.h"
#include "Ply.h"
#include "MemoryUsage.h"
#include "Octree.h"
#include "MultiGridOctreeData.h"

#pragma mark - Internal Helpers

template<class Real>
struct _ColorInfo
{
    static Point3D<Real> ReadASCII(FILE *fp)
    {
        Point3D<unsigned char> c;
        if (fscanf(fp, " %c %c %c ", &c[0], &c[1], &c[2]) != 3)
        {
            NSLog(@"[ERROR] Failed to read color\n");
            exit(0);
        }
        
        return Point3D<Real>((Real)c[0], (Real)c[1], (Real)c[2]);
    };
    
    static bool ValidPlyProperties(const bool *props) {
        return (props[0] || props[3]) && (props[1] || props[4]) && (props[2] || props[5]);
    }
    
    const static PlyProperty PlyProperties[];
};

template<>
const PlyProperty _ColorInfo<float>::PlyProperties[] =
{
    { (char *)"r"    , PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[0])), 0, 0, 0, 0 },
    { (char *)"g"    , PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[1])), 0, 0, 0, 0 },
    { (char *)"b"    , PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[2])), 0, 0, 0, 0 },
    { (char *)"red"  , PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[0])), 0, 0, 0, 0 },
    { (char *)"green", PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[1])), 0, 0, 0, 0 },
    { (char *)"blue" , PLY_UCHAR, PLY_FLOAT, int(offsetof(Point3D<float>, coords[2])), 0, 0, 0, 0 },
};

template<class Real>
XForm4x4<Real> GetPointXForm(OrientedPointStream<Real> &stream, Real scaleFactor)
{
    Point3D<Real> min, max;
    stream.boundingBox(min, max);
    Point3D<Real> center = (max + min) / 2;
    Real scale = std::max<Real>(max[0] - min[0], std::max<Real>(max[1] - min[1], max[2] - min[2]));
    scale *= scaleFactor;
    
    for (int i = 0; i < 3; i++)
    {
        center[i] -= scale / 2;
    }
    
    XForm4x4<Real> tXForm = XForm4x4<Real>::Identity(), sXForm = XForm4x4<Real>::Identity();
    
    for (int i = 0 ; i < 3; i++)
    {
        sXForm(i, i) = (Real)(1.0 / scale);
        tXForm(3, i) = -center[i];
    }
    
    return sXForm * tXForm;
}

#pragma mark - Execute

template<class Real, class Vertex>
int _Execute(NSString *inputFilePath, NSString *outputFilePath)
{
    const int   AdaptiveExponent = 1;
    const bool  ASCII = true;
    const BoundaryType BoundType = BOUNDARY_NEUMANN;
    const int   CGDepth = 0;
    const float CGSolverAccuracy = 1e-3f;
    const float Color = 16.0f;
    const bool  Confidence = false;
    const int   Degree = 2;
    const bool  Density = false;
    const int   Depth = 8;
    const int   FullDepth = 5;
    const int   Iters = 8;
    const int   KernelDepth = Depth - 2;
    const bool  LinearFit = false;
    const int   LowResIterMultiplier = 1;
    const int   MaxSolveDepth = Depth;
    const bool  NonManifold = false;
    const float SamplesPerNode = 1.0f;
    const float Scale = 1.1f;
    const float PointWeight = 4.0f;
    const bool  PolygonMesh = false;
    const bool  ShowResidual = false;
    const int   ThreadCount = omp_get_num_procs();
    const bool  Verbose = true;
    
    const char *In = [inputFilePath cStringUsingEncoding:NSASCIIStringEncoding];
    const char *Out = [outputFilePath cStringUsingEncoding:NSASCIIStringEncoding];
    
    typedef typename Octree<Real>::template DensityEstimator<WEIGHT_DEGREE> DensityEstimator;
    typedef typename Octree<Real>::template InterpolationInfo<false> InterpolationInfo;
    typedef OrientedPointStream<Real> PointStream;
    typedef OrientedPointStreamWithData<Real, Point3D<Real>> PointStreamWithData;
    typedef TransformedOrientedPointStream<Real> XPointStream;
    typedef TransformedOrientedPointStreamWithData<Real, Point3D<Real>> XPointStreamWithData;
    Reset<Real>();
    
    NSLog(@"Running Screened Poisson Reconstruction (Version 9.011)\n");
    
    double startTime = Time();
    
    Octree<Real> tree;
    tree.threads = ThreadCount;
    OctNode<TreeNodeData>::SetAllocator(MEMORY_ALLOCATOR_BLOCK_SIZE);
    
    Real isoValue = 0;
    int pointCount = 0;
    Real pointWeightSum = 0;
    std::vector<typename Octree<Real>::PointSample> *samples = new std::vector<typename Octree<Real>::PointSample>();
    std::vector<ProjectiveData<Point3D<Real>, Real>> *sampleData = NULL;
    DensityEstimator *densityEstimator = NULL;
    SparseNodeData<Point3D<Real>, NORMAL_DEGREE> *normalInfo = NULL;
    Real targetValue = (Real)0.5;
    XForm4x4<Real> transform = XForm4x4<Real>::Identity();
    XForm4x4<Real> inverseTransform = transform.inverse();
    
    // Read in the samples (and color data)
    {
        PointStream *pointStream;
        const char *ext = [[inputFilePath pathExtension] cStringUsingEncoding:NSASCIIStringEncoding];
        if (Color > 0)
        {
            sampleData = new std::vector<ProjectiveData<Point3D<Real>, Real>>();
            if (!strcasecmp(ext, "bnpts"))
            {
                pointStream = new BinaryOrientedPointStreamWithData<Real, Point3D<Real>, float, Point3D<unsigned char>>(In);
            }
            else if (!strcasecmp(ext, "ply"))
            {
                pointStream = new PLYOrientedPointStreamWithData<Real, Point3D<Real>>(In, _ColorInfo<Real>::PlyProperties, 6, _ColorInfo<Real>::ValidPlyProperties);
            }
            else
            {
                pointStream = new ASCIIOrientedPointStreamWithData<Real, Point3D<Real>>(In, _ColorInfo<Real>::ReadASCII);
            }
        }
        else
        {
            if (!strcasecmp(ext, "bnpts"))
            {
                pointStream = new BinaryOrientedPointStream<Real, float>(In);
            }
            else if (!strcasecmp(ext, "ply"))
            {
                pointStream = new PLYOrientedPointStream<Real>(In);
            }
            else
            {
                pointStream = new ASCIIOrientedPointStream<Real>(In);
            }
        }
        
        XPointStream _pointStream(transform, *pointStream);
        transform = GetPointXForm(_pointStream, (Real)Scale) * transform;
        if (sampleData)
        {
            XPointStreamWithData _pointStream(transform, (PointStreamWithData &)*pointStream);
            pointCount = tree.template init<Point3D<Real>>(_pointStream, Depth, Confidence, *samples, sampleData);
        }
        else
        {
            XPointStream _pointStream(transform, *pointStream);
            pointCount = tree.template init<Point3D<Real>>(_pointStream, Depth, Confidence, *samples, sampleData);
        }
        inverseTransform = transform.inverse();
        delete pointStream;
#pragma omp parallel for num_threads(Threads)
        for (int i = 0; i < (int)samples->size(); i++)
        {
            (*samples)[i].sample.data.n *= (Real) - 1;
        }
        
        NSLog(@"Input Points / Samples: %d / %d\n", pointCount, (int)samples->size());
    }
    
    DenseNodeData<Real, Degree> solution;
    
    {
        DenseNodeData<Real, Degree> constraints;
        InterpolationInfo *interpolationInfo = NULL;
        int solveDepth = MaxSolveDepth;
        
        tree.resetNodeIndices();
        
        // Get the kernel density estimator [If discarding, compute anew. Otherwise, compute once.]
        {
            densityEstimator = tree.template setDensityEstimator<WEIGHT_DEGREE>(*samples, KernelDepth, SamplesPerNode);
        }
        
        // Transform the Hermite samples into a vector field [If discarding, compute anew. Otherwise, compute once.]
        {
            normalInfo = new SparseNodeData<Point3D<Real>, NORMAL_DEGREE>();
            *normalInfo = tree.template setNormalField<NORMAL_DEGREE>(*samples, *densityEstimator, pointWeightSum, true);
        }
        
        if (!Density) { delete densityEstimator; densityEstimator = NULL; }
        
        // Trim the tree and prepare for multigrid
        {
            std::vector<int> indexMap;
            
            constexpr int MAX_DEGREE = NORMAL_DEGREE> Degree ? NORMAL_DEGREE : Degree;
            tree.template inalizeForBroodedMultigrid<MAX_DEGREE, Degree, BoundType>(FullDepth, typename Octree<Real>::template HasNormalDataFunctor<NORMAL_DEGREE>(*normalInfo), &indexMap);
            
            if (normalInfo) { normalInfo->remapIndices(indexMap); }
            if (densityEstimator) { densityEstimator->remapIndices(indexMap); }
        }
        
        // Add the FEM constraints
        if (samples->size() > 0)
        {
            constraints = tree.template initDenseNodeData<Degree>();
            tree.template addFEMConstraints<Degree, BoundType, NORMAL_DEGREE, BoundType>(FEMVFConstraintFunctor<NORMAL_DEGREE, BoundType, Degree, BoundType>(1.0, 0.0), *normalInfo, constraints, solveDepth);
        }
        
        // Free up the normal info [If we don't need it for subseequent iterations.]
        if (normalInfo) { delete normalInfo; normalInfo = NULL; }
        
        // Add the interpolation constraints
        if (PointWeight > 0)
        {
            interpolationInfo = new InterpolationInfo(tree, *samples, targetValue, AdaptiveExponent, (Real)PointWeight * pointWeightSum, (Real)0);
            tree.template addInterpolationConstraints<Degree, BoundType>(*interpolationInfo, constraints, solveDepth);
        }
        
        NSLog(@"Leaf Nodes / Active Nodes / Ghost Nodes: %d / %d / %d\n", (int)tree.leaves(), (int)tree.nodes(), (int)tree.ghostNodes());
#if !TARGET_OS_IOS
        NSLog(@"Memory Usage: %.3f MB\n", float(MemoryInfo::Usage())/(1<<20));
#endif
        
        // Solve the linear system
        if (samples->size() > 0)
        {
            typename Octree<Real>::SolverInfo solverInfo;
            solverInfo.cgDepth = CGDepth;
            solverInfo.iters = Iters;
            solverInfo.cgAccuracy = CGSolverAccuracy;
            solverInfo.verbose = Verbose;
            solverInfo.showResidual = ShowResidual;
            solverInfo.lowResIterMultiplier = std::max<double>(1.0, LowResIterMultiplier);
            solution = tree.template solveSystem<Degree, BoundType>(FEMSystemFunctor<Degree, BoundType>(0, 1, 0), interpolationInfo, constraints, solveDepth, solverInfo);
            
            if (interpolationInfo) { delete interpolationInfo; interpolationInfo = NULL; }
        }
    }
    
    char tempHeader[1024];
    {
        const char FileSeparator = '/';
        const char *tempPath = [NSTemporaryDirectory() cStringUsingEncoding:NSASCIIStringEncoding];
        if (tempPath[strlen(tempPath) - 1] == FileSeparator)
        {
            sprintf(tempHeader, "%sPR_", tempPath);
        }
        else
        {
            sprintf(tempHeader, "%s%cPR_", tempPath, FileSeparator);
        }
    }
    CoredFileMeshData<Vertex> mesh(tempHeader);
    
    if (samples->size() > 0)
    {
        double valueSum = 0, weightSum = 0;
        typename Octree<Real>::template MultiThreadedEvaluator<Degree, BoundType> evaluator(&tree, solution, ThreadCount);
        
#pragma omp parallel for num_threads(Threads) reduction(+ : valueSum, weightSum)
        for (size_t j = 0; j < samples->size(); j++)
        {
            ProjectiveData<OrientedPoint3D<Real>, Real> &sample = (*samples)[j].sample;
            Real w = sample.weight;
            if (w > 0)
            {
                weightSum += w;
                valueSum += evaluator.value(sample.data.p / sample.weight, omp_get_thread_num(), (*samples)[j].node) * w;
            }
        }
        
        isoValue = (Real)(valueSum / weightSum);
        if (!(Color > 0) && samples) { delete samples; samples = NULL; }
        NSLog(@"Iso-Value: %e\n", isoValue);
    }
    
    SparseNodeData<ProjectiveData<Point3D<Real>, Real>, DATA_DEGREE> *colorData = NULL;
    
    if (sampleData)
    {
        colorData = new SparseNodeData<ProjectiveData<Point3D<Real>, Real>, DATA_DEGREE>();
        *colorData = tree.template setDataField<DATA_DEGREE, false>(*samples, *sampleData, (DensityEstimator *)NULL);
        delete sampleData;
        sampleData = NULL;
        
        for (const OctNode<TreeNodeData> *n = tree.tree().nextNode(); n; n = tree.tree().nextNode(n))
        {
            ProjectiveData<Point3D<Real>, Real> *color = (*colorData)(n);
            if (color)
            {
                (*color) *= (Real)pow(Color, tree.depth(n));
            }
        }
    }
    
    tree.template getMCIsoSurface<Degree, BoundType, WEIGHT_DEGREE, DATA_DEGREE>(densityEstimator, colorData, solution, isoValue, mesh, !LinearFit, !NonManifold, PolygonMesh);
    
    NSLog(@"Vertices / Polygons: %d / %d\n", mesh.outOfCorePointCount() + (int)mesh.inCorePoints.size(), mesh.polygonCount());
    if (colorData) { delete colorData, colorData = NULL; }
    
    if (ASCII)
    {
        PlyWritePolygons((char *)Out, &mesh, PLY_ASCII, NULL, 0, inverseTransform);
    }
    else
    {
        PlyWritePolygons((char *)Out, &mesh, PLY_BINARY_NATIVE, NULL, 0, inverseTransform);
    }
    
    if (densityEstimator) { delete densityEstimator; densityEstimator = NULL; }
    NSLog(@"Total Solve: %9.1f (s), %9.1f (MB)\n", Time() - startTime, tree.maxMemoryUsage());
    
    return 1;
}

#pragma mark - Exported

int Execute(NSString *inputFilePath, NSString *outputFilePath)
{
    return _Execute<float, PlyVertex<float>>(inputFilePath, outputFilePath);
}
