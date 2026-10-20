/*========================== begin_copyright_notice ============================

Copyright (C) 2019-2026 Intel Corporation

SPDX-License-Identifier: MIT

============================= end_copyright_notice ===========================*/

//     File Name:  metrics_discovery_api.h

//     Abstract:   Interface for metrics discovery DLL

#pragma once

#include <stdint.h>

#ifdef _MSC_VER
    #define MD_STDCALL __stdcall
#else
    #define MD_STDCALL
#endif // _MSC_VER

//////////////////////////////////////////////////////////////////////////////////
// Helper macro to check required API version.
// Combines major and minor into one, comparable 64bit value.
//////////////////////////////////////////////////////////////////////////////////
#define MD_API_VERSION_COMBINE_MAJOR_MINOR( version ) \
    ( (uint64_t) ( version ).MajorNumber << 32 | (uint64_t) ( version ).MinorNumber )

//////////////////////////////////////////////////////////////////////////////////
// Macro to check required API version.
// Uses TApiVersion_1_0 struct.
//////////////////////////////////////////////////////////////////////////////////
#define MD_API_VERSION_AT_LEAST( requiredVersion, currentVersion ) \
    ( MD_API_VERSION_COMBINE_MAJOR_MINOR( ( currentVersion ) ) > MD_API_VERSION_COMBINE_MAJOR_MINOR( ( requiredVersion ) ) || MD_API_VERSION_COMBINE_MAJOR_MINOR( ( currentVersion ) ) == MD_API_VERSION_COMBINE_MAJOR_MINOR( ( requiredVersion ) ) && ( currentVersion ).BuildNumber >= ( requiredVersion ).BuildNumber )

//////////////////////////////////////////////////////////////////////////////////
// API build number:
//////////////////////////////////////////////////////////////////////////////////
#define MD_API_BUILD_NUMBER_CURRENT 197

namespace MetricsDiscovery
{
    //////////////////////////////////////////////////////////////////////////////////
    // API major version number:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMD_API_MAJOR_VERSION
    {
        MD_API_MAJOR_NUMBER_1       = 1,
        MD_API_MAJOR_NUMBER_CURRENT = MD_API_MAJOR_NUMBER_1,
        MD_API_MAJOR_NUMBER_CEIL    = 0xFFFFFFFF
    } MD_API_MAJOR_VERSION;

    //////////////////////////////////////////////////////////////////////////////////
    // API minor version number:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMD_API_MINOR_VERSION
    {
        MD_API_MINOR_NUMBER_0       = 0,
        MD_API_MINOR_NUMBER_1       = 1,  // CalculationAPI
        MD_API_MINOR_NUMBER_2       = 2,  // OverridesAPI
        MD_API_MINOR_NUMBER_3       = 3,  // BatchBuffer Sampling (aka DMA Sampling)
        MD_API_MINOR_NUMBER_4       = 4,  // GT dependent MetricSets
        MD_API_MINOR_NUMBER_5       = 5,  // MaxValue calculation for CalculationAPI
        MD_API_MINOR_NUMBER_6       = 6,  // Multi adapter support
        MD_API_MINOR_NUMBER_7       = 7,  // Compile time equations calculation approach
        MD_API_MINOR_NUMBER_8       = 8,  // TAdapterParams update
        MD_API_MINOR_NUMBER_9       = 9,  // Sub device support.
        MD_API_MINOR_NUMBER_10      = 10, // GetGpuCpuTimestamps API function extended by a correlation indicator param
        MD_API_MINOR_NUMBER_11      = 11, // Add availability equations for metric sets
        MD_API_MINOR_NUMBER_12      = 12, // Add support for Information Set in concurrent group
        MD_API_MINOR_NUMBER_13      = 13, // Extend API to support flexible metric sets
        MD_API_MINOR_NUMBER_14      = 14, // Offline calculation support
        MD_API_MINOR_NUMBER_15      = 15, // Change IO Stream state
        MD_API_MINOR_NUMBER_16      = 16, // Metrics Aggregation Support
        MD_API_MINOR_NUMBER_CURRENT = MD_API_MINOR_NUMBER_16,
        MD_API_MINOR_NUMBER_CEIL    = 0xFFFFFFFF
    } MD_API_MINOR_VERSION;

    //////////////////////////////////////////////////////////////////////////////////
    // Completion codes:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum ECompletionCode
    {
        CC_OK                      = 0,
        CC_READ_PENDING            = 1,
        CC_ALREADY_INITIALIZED     = 2,
        CC_STILL_INITIALIZED       = 3,
        CC_CONCURRENT_GROUP_LOCKED = 4,
        CC_WAIT_TIMEOUT            = 5,
        CC_TRY_AGAIN               = 6,
        CC_INTERRUPTED             = 7,
        CC_NOT_ENOUGH_DATA         = 8,
        CC_NO_RESULT               = 9,
        CC_ERROR_INVALID_PARAMETER = 40,
        CC_ERROR_NO_MEMORY         = 41,
        CC_ERROR_GENERAL           = 42,
        CC_ERROR_FILE_NOT_FOUND    = 43,
        CC_ERROR_NOT_SUPPORTED     = 44,
        CC_ERROR_ACCESS_DENIED     = 45,
        CC_LAST_1_0                = 46
    } TCompletionCode;

    /* Forward declarations */

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the adapter group object.
    //////////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_6;
    class IAdapterGroup_1_8;
    class IAdapterGroup_1_9;
    class IAdapterGroup_1_10;
    class IAdapterGroup_1_11;
    class IAdapterGroup_1_13;
    class IAdapterGroup_1_14;
    class IAdapterGroup_1_15;
    class IAdapterGroup_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the GPU adapter object.
    //////////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_6;
    class IAdapter_1_8;
    class IAdapter_1_9;
    class IAdapter_1_10;
    class IAdapter_1_11;
    class IAdapter_1_13;
    class IAdapter_1_15;
    class IAdapter_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the GPU metrics root object.
    //////////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_0;
    class IMetricsDevice_1_1;
    class IMetricsDevice_1_2;
    class IMetricsDevice_1_5;
    class IMetricsDevice_1_10;
    class IMetricsDevice_1_11;
    class IMetricsDevice_1_13;
    class IMetricsDevice_1_15;
    class IMetricsDevice_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for Metrics Device overrides.
    //////////////////////////////////////////////////////////////////////////////////
    class IOverride_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metrics groups that can be collected concurrently
    // to another group.
    //////////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_0;
    class IConcurrentGroup_1_1;
    class IConcurrentGroup_1_3;
    class IConcurrentGroup_1_5;
    class IConcurrentGroup_1_11;
    class IConcurrentGroup_1_13;
    class IConcurrentGroup_1_15;
    class IConcurrentGroup_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metric sets mapping to different HW configuration
    // that should be used exclusively to each other metric set in the concurrent
    // group.
    //////////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_0;
    class IMetricSet_1_1;
    class IMetricSet_1_4;
    class IMetricSet_1_5;
    class IMetricSet_1_11;
    class IMetricSet_1_13;
    class IMetricSet_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metric that is sampled.
    //////////////////////////////////////////////////////////////////////////////////
    class IMetric_1_0;
    class IMetric_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the measurement information (report reason, etc.).
    //////////////////////////////////////////////////////////////////////////////////
    class IInformation_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metric read and normalization equation.
    //////////////////////////////////////////////////////////////////////////////////
    class IEquation_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metric enumerator object.
    //////////////////////////////////////////////////////////////////////////////////
    class IMetricEnumerator_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the metric prototype object.
    //////////////////////////////////////////////////////////////////////////////////
    class IMetricPrototype_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Abstract interface for the calculation context object.
    //////////////////////////////////////////////////////////////////////////////////
    class ICalculationContext_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Value types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EValueType
    {
        VALUE_TYPE_UINT32,
        VALUE_TYPE_UINT64,
        VALUE_TYPE_FLOAT,
        VALUE_TYPE_BOOL,
        VALUE_TYPE_CSTRING,
        VALUE_TYPE_BYTEARRAY,
        VALUE_TYPE_UINT32_RANGE,
        VALUE_TYPE_UINT64_RANGE,
        // ...
        VALUE_TYPE_LAST,
    } TValueType;

    //////////////////////////////////////////////////////////////////////////////////
    // Byte Array:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SByteArray_1_0
    {
        uint32_t Size;
        uint8_t* Data;
    } TByteArray_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Typed value:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct STypedValue_1_0
    {
        TValueType ValueType;
        union
        {
            uint32_t ValueUInt32;
            uint64_t ValueUInt64;
            struct
            {
                uint32_t Low;
                uint32_t High;
            } ValueUInt64Fields;
            float           ValueFloat;
            bool            ValueBool;
            char*           ValueCString;
            TByteArray_1_0* ValueByteArray; // Dynamically allocated
        };
    } TTypedValue_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Global symbol:
    //     Global symbols will be available to describe SKU specific information.
    //     Example global symbols:
    //     "EuCoresTotalCount", "EuThreadsCount", "EuSlicesTotalCount", "EuSubslicesTotalCount",
    //     "SamplersTotalCount", "PciDeviceId", "NumberOfShadingUnits", "GpuTimestampFrequency",
    //     "MaxTimestamp", "GpuMinFrequencyMHz", "GpuMaxFrequencyMHz"
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SGlobalSymbol_1_0
    {
        const char*     SymbolName;
        TTypedValue_1_0 SymbolTypedValue;
    } TGlobalSymbol_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // API version:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SApiVersion_1_0
    {
        uint32_t MajorNumber;
        uint32_t MinorNumber;
        uint32_t BuildNumber;
    } TApiVersion_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metrics Device:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricsDeviceParams_1_0
    {
        // API version
        TApiVersion_1_0 Version;
        uint32_t        ConcurrentGroupsCount;
        uint32_t        GlobalSymbolsCount;
        uint32_t        DeltaFunctionsCount;
        uint32_t        EquationElementTypesCount;
        uint32_t        EquationOperationsCount;
        const char*     DeviceName;
    } TMetricsDeviceParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metrics Device 1.2:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricsDeviceParams_1_2 : public SMetricsDeviceParams_1_0
    {
        uint32_t OverrideCount;
    } TMetricsDeviceParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Metric API types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMetricApiType
    {
        API_TYPE_IOSTREAM = 0x00000001, // API independent method
        API_TYPE_DX9      = 0x00000002,
        API_TYPE_DX10     = 0x00000004,
        API_TYPE_DX11     = 0x00000008,
        API_TYPE_OGL      = 0x00000010,
        API_TYPE_OGL4_X   = 0x00000020,
        API_TYPE_OCL      = 0x00000040,
        API_TYPE_MEDIA    = 0x00000080, // Obsolete
        API_TYPE_DX12     = 0x00000100,
        API_TYPE_BBSTREAM = 0x00000200, // Obsolete
        API_TYPE_VULKAN   = 0x00000400,
        API_TYPE_RESERVED = 0x00000800,
        API_TYPE_ALL      = 0xFFFFFFFF
    } TMetricApiType;

    //////////////////////////////////////////////////////////////////////////////////
    // Measurement types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMeasurementType
    {
        MEASUREMENT_TYPE_SNAPSHOT_IO    = 0x00000001,
        MEASUREMENT_TYPE_SNAPSHOT_QUERY = 0x00000002,
        MEASUREMENT_TYPE_DELTA_QUERY    = 0x00000004,
        MEASUREMENT_TYPE_ALL            = 0x0000FFFF,
    } TMeasurementType;

    //////////////////////////////////////////////////////////////////////////////////
    // Usage flags:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMetricUsageFlag
    {
        USAGE_FLAG_OVERVIEW  = 0x00000001, // GPU system overview metric, useful for high level workload characterization
        USAGE_FLAG_INDICATE  = 0x00000002, // Metric indicating a performance problem, useful when comparing with threshold
        USAGE_FLAG_CORRELATE = 0x00000004, // Metric correlating with performance problem, useful for proving to false only
        USAGE_FLAG_SYSTEM    = 0x00000020, // Metric useful at system level
        USAGE_FLAG_FRAME     = 0x00000040, // Metric useful at frame level
        USAGE_FLAG_BATCH     = 0x00000080, // Metric useful at batch level
        USAGE_FLAG_DRAW      = 0x00000100, // Metric useful at draw level
        USAGE_FLAG_TIER_1    = 0x00000400,
        USAGE_FLAG_TIER_2    = 0x00000800,
        USAGE_FLAG_TIER_3    = 0x00001000,
        USAGE_FLAG_TIER_4    = 0x00002000,
        USAGE_FLAG_GLASS_JAW = 0x00004000,
        USAGE_FLAG_ALL       = 0x0000FFFF,
    } TMetricUsageFlag;

    //////////////////////////////////////////////////////////////////////////////////
    // Sampling types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum ESamplingType
    {
        SAMPLING_TYPE_OA_TIMER     = 0x00000001,
        SAMPLING_TYPE_OA_EVENT     = 0x00000002, // Obsolete
        SAMPLING_TYPE_GPU_QUERY    = 0x00000004, // Obsolete
        SAMPLING_TYPE_DMA_BUFFER   = 0x00000008, // Obsolete
        SAMPLING_TYPE_OAM_TIMER    = 0x00000010,
        SAMPLING_TYPE_OAMERT_TIMER = 0x00000020,
        SAMPLING_TYPE_ALL          = 0x0000FFFF,
    } TSamplingType;

    //////////////////////////////////////////////////////////////////////////////////
    // Metric categories:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMetricCategory
    {
        GPU_RENDER  = 0x0001,
        GPU_COMPUTE = 0x0002,
        GPU_MEDIA   = 0x0004,
        GPU_GENERIC = 0x0008, // Does not belong to any specific category like memory traffic
    } TMetricCategory;

    //////////////////////////////////////////////////////////////////////////////////
    // IoStream read flags:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EIoReadFlag
    {
        IO_READ_FLAG_DROP_OLD_REPORTS    = 0x00000001,
        IO_READ_FLAG_GET_CONTEXT_ID_TAGS = 0x00000002,
    } TIoReadFlag;

    //////////////////////////////////////////////////////////////////////////////////
    // Override modes:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EOverrideMode
    {
        OVERRIDE_MODE_GLOBAL = 0x0001,
        OVERRIDE_MODE_LOCAL  = 0x0002,
    } TOverrideMode;

    //////////////////////////////////////////////////////////////////////////////////
    // Adapter capability flags:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EAdapterCapability
    {
        ADAPTER_CAPABILITY_UNDEFINED        = 0,
        ADAPTER_CAPABILITY_RENDER_SUPPORTED = 1 << 0,
    } TAdapterCapability;

    //////////////////////////////////////////////////////////////////////////////////
    // Adapter types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EAdapterType
    {
        ADAPTER_TYPE_UNDEFINED = 0,
        ADAPTER_TYPE_INTEGRATED,
        ADAPTER_TYPE_DISCRETE,
    } TAdapterType;

    //////////////////////////////////////////////////////////////////////////////////
    // Adapter ID types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EAdapterIdType
    {
        ADAPTER_ID_TYPE_UNDEFINED = 0,
        ADAPTER_ID_TYPE_LUID,
        ADAPTER_ID_TYPE_MAJOR_MINOR,
    } TAdapterIdType;

    //////////////////////////////////////////////////////////////////////////////////
    // LUID (locally unique identifier) adapter ID:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterIdLuid_1_6
    {
        uint32_t LowPart;
        int32_t  HighPart;
    } TAdapterIdLuid_1_6;

    //////////////////////////////////////////////////////////////////////////////////
    // Major / minor pair adapter ID:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterIdMajorMinor_1_6
    {
        int32_t Major;
        int32_t Minor;
    } TAdapterIdMajorMinor_1_6;

    //////////////////////////////////////////////////////////////////////////////////
    // Adapter ID:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterId_1_6
    {
        TAdapterIdType Type;
        union
        {
            TAdapterIdLuid_1_6       Luid;
            TAdapterIdMajorMinor_1_6 MajorMinor;
        };
    } TAdapterId_1_6;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU adapter:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterParams_1_6
    {
        const char*    ShortName;
        TAdapterId_1_6 SystemId; // Operating system specific adapter id
        uint32_t       VendorId;
        uint32_t       SubVendorId;
        uint32_t       DeviceId;
        uint32_t       Platform;
        uint32_t       BusNumber;
        uint32_t       DeviceNumber;
        uint32_t       FunctionNumber;
        TAdapterType   Type;           // Adapter type, e.g. integrated, discrete
        uint32_t       CapabilityMask; // Consists of TAdapterCapability flags, e.g. RENDER_SUPPORTED
    } TAdapterParams_1_6;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU adapter:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterParams_1_8 : public SAdapterParams_1_6
    {
        uint32_t DomainNumber;
    } TAdapterParams_1_8;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU adapter:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterParams_1_9 : public SAdapterParams_1_8
    {
        uint32_t SubDevicesCount;
    } TAdapterParams_1_9;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU sub device:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SSubDeviceParams_1_9
    {
        uint32_t EnginesCount;
    } TSubDeviceParams_1_9;

    //////////////////////////////////////////////////////////////////////////////////
    // Engine ID types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EEngineIdType
    {
        ENGINE_ID_TYPE_CLASS_INSTANCE = 0
    } TEngineIdType;

    //////////////////////////////////////////////////////////////////////////////////
    // Class / instance engine ID:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SEngineIdClassInstance_1_9
    {
        uint32_t Class;
        uint32_t Instance;
    } TEngineIdClassInstance_1_9;

    //////////////////////////////////////////////////////////////////////////////////
    // Engine identification:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SEngineId_1_9
    {
        TEngineIdType Type;

        union
        {
            TEngineIdClassInstance_1_9 ClassInstance;
        };
    } TEngineId_1_9;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU engine:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SEngineParams_1_9
    {
        TEngineId_1_9 EngineId;
    } TEngineParams_1_9;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of GPU engine:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SEngineParams_1_13 : public SEngineParams_1_9
    {
        uint32_t GtId;
        uint32_t OaUnit;
    } TEngineParams_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Adapter Group:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SAdapterGroupParams_1_6
    {
        // API version
        TApiVersion_1_0 Version;
        uint32_t        AdapterCount;
    } TAdapterGroupParams_1_6;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Concurrent Group:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SConcurrentGroupParams_1_0
    {
        const char* SymbolName;  // For example "OA" or "OAM0" or "PipeStats"
        const char* Description; // For example "OA Unit Metrics"
        uint32_t    MeasurementTypeMask;
        uint32_t    MetricSetsCount;
        uint32_t    IoMeasurementInformationCount;
        uint32_t    IoGpuContextInformationCount;

    } TConcurrentGroupParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Concurrent Group:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SConcurrentGroupParams_1_13 : SConcurrentGroupParams_1_0
    {
        const char* StreamEventName;
    } TConcurrentGroupParams_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of an Override:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SOverrideParams_1_2
    {
        const char* SymbolName;  // For example "FrequencyOverride"
        const char* Description; // For example "Overrides device GPU frequency with a static value."
        uint32_t    ApiMask;
        uint32_t    PlatformMask;
        uint32_t    OverrideModeMask;

    } TOverrideParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Base params of SetOverride method:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SSetOverrideParams_1_2
    {
        bool Enable;
    } TSetOverrideParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Frequency override specific SetOverride params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SSetFrequencyOverrideParams_1_2 : SSetOverrideParams_1_2
    {
        uint32_t FrequencyMhz;
        uint32_t Pid;
    } TSetFrequencyOverrideParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Query override specific SetOverride params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SSetQueryOverrideParams_1_2 : SSetOverrideParams_1_2
    {
        uint32_t Period; // Nanoseconds
    } TSetQueryOverrideParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // Driver override params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SSetDriverOverrideParams_1_2 : SSetOverrideParams_1_2
    {
        uint32_t Value;
    } TSetDriverOverrideParams_1_2;

    //////////////////////////////////////////////////////////////////////////////////
    // API specific id:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SApiSpecificId_1_0
    {
        uint32_t    D3D9QueryId;           // D3D9 Query ID
        uint32_t    D3D9Fourcc;            // D3D9 FourCC
        uint32_t    D3D1XQueryId;          // D3D1X Query ID
        uint32_t    D3D1XDevDependentId;   // D3D1X device dependent counter ID
        const char* D3D1XDevDependentName; // Device dependent counter name
        uint32_t    OGLQueryIntelId;       // Intel OGL query extension ID
        const char* OGLQueryIntelName;     // Intel OGL query extension name
        uint32_t    OGLQueryARBTargetId;   // ARB OGL Query Target ID
        uint32_t    OCL;                   // OCL configuration ID
        uint32_t    HwConfigId;            // Config ID for IO stream
        uint32_t    placeholder[1];
    } TApiSpecificId_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metric set:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricSetParams_1_0
    {
        const char*        SymbolName; // For example "Dx11Tessellation"
        const char*        ShortName;  // For example "DX11 Tessellation Metrics Set"
        uint32_t           ApiMask;
        uint32_t           CategoryMask;
        uint32_t           RawReportSize;   // As in HW
        uint32_t           QueryReportSize; // As in Query API
        uint32_t           MetricsCount;
        uint32_t           InformationCount;
        uint32_t           ComplementarySetsCount;
        TApiSpecificId_1_0 ApiSpecificId;
        uint32_t           PlatformMask;
    } TMetricSetParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // GT differenced MetricSet params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricSetParams_1_4 : SMetricSetParams_1_0
    {
        uint32_t GtMask;
    } TMetricSetParams_1_4;

    //////////////////////////////////////////////////////////////////////////////////
    // AvailabilityEquation differenced MetricSet params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricSetParams_1_11 : SMetricSetParams_1_4
    {
        const char* AvailabilityEquation;
    } TMetricSetParams_1_11;

    //////////////////////////////////////////////////////////////////////////////////
    // Raw counter size and count of Metric set:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricSetParams_1_16 : SMetricSetParams_1_11
    {
        uint32_t RawCounterSize;
        uint32_t RawCounterCount;
    } TMetricSetParams_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Metric result types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMetricResultType
    {
        RESULT_UINT32,
        RESULT_UINT64,
        RESULT_BOOL,
        RESULT_FLOAT,
        RESULT_LAST
    } TMetricResultType;

    //////////////////////////////////////////////////////////////////////////////////
    // Metric types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EMetricType
    {
        METRIC_TYPE_DURATION,
        METRIC_TYPE_EVENT,
        METRIC_TYPE_EVENT_WITH_RANGE,
        METRIC_TYPE_THROUGHPUT,
        METRIC_TYPE_TIMESTAMP,
        METRIC_TYPE_FLAG,
        METRIC_TYPE_RATIO,
        METRIC_TYPE_RAW,
        METRIC_TYPE_LAST
    } TMetricType;

    //////////////////////////////////////////////////////////////////////////////////
    // Information types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EInformationType
    {
        INFORMATION_TYPE_REPORT_REASON,
        INFORMATION_TYPE_VALUE,
        INFORMATION_TYPE_FLAG,
        INFORMATION_TYPE_TIMESTAMP,
        INFORMATION_TYPE_CONTEXT_ID_TAG,
        INFORMATION_TYPE_SAMPLE_PHASE,
        INFORMATION_TYPE_GPU_NODE,
        INFORMATION_TYPE_LAST
    } TInformationType;

    //////////////////////////////////////////////////////////////////////////////////
    // Report reasons:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EReportReason
    {
        REPORT_REASON_UNDEFINED                 = 0x0000,
        REPORT_REASON_INTERNAL_TIMER            = 0x0001,
        REPORT_REASON_INTERNAL_TRIGGER1         = 0x0002,
        REPORT_REASON_INTERNAL_TRIGGER2         = 0x0004,
        REPORT_REASON_INTERNAL_CONTEXT_SWITCH   = 0x0008,
        REPORT_REASON_INTERNAL_GO               = 0x0010,
        REPORT_REASON_INTERNAL_FREQUENCY_CHANGE = 0x0020,
        REPORT_REASON_INTERNAL_MMIO_TRIGGER     = 0x0040,
        REPORT_REASON_QUERY_DEFAULT             = 0x0100,
        REPORT_REASON_QUERY_INTERNAL_RESOLVE    = 0x0200,
        REPORT_REASON_QUERY_INTERNAL_CLEAR      = 0x0400,
    } TReportReason;

    //////////////////////////////////////////////////////////////////////////////////
    // Sample phase:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum ESamplePhase
    {
        SAMPLE_PHASE_END,
        SAMPLE_PHASE_BEGIN,
        SAMPLE_PHASE_LAST
    } TSamplePhase;

    //////////////////////////////////////////////////////////////////////////////////
    // Gpu Node:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EInformationGpuNode
    {
        INFORMATION_GPUNODE_3D       = 0, // Available by default on all platform
        INFORMATION_GPUNODE_VIDEO    = 1, // Available on CTG+
        INFORMATION_GPUNODE_BLT      = 2, // Available on GT
        INFORMATION_GPUNODE_VE       = 3, // Available on HSW+ (VideoEnhancement)
        INFORMATION_GPUNODE_VCS2     = 4, // Available on BDW+ GT3+
        INFORMATION_GPUNODE_REAL_MAX = 5, // All nodes beyond this are virtual nodes - they don't have an actual GPU engine
        INFORMATION_GPUNODE_LAST
    } TInformationGpuNode;

    //////////////////////////////////////////////////////////////////////////////////
    // Hardware unit types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EHwUnitType
    {
        HW_UNIT_GPU,
        HW_UNIT_SLICE,
        HW_UNIT_SUBSLICE,
        HW_UNIT_SUBSLICE_BANK,
        HW_UNIT_EU_UNIT,
        HW_UNIT_UNCORE,
        HW_UNIT_DUALSUBSLICE,
        HW_UNIT_LAST
    } THwUnitType;

    //////////////////////////////////////////////////////////////////////////////////
    // Delta function types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EDeltaFunctionType
    {
        DELTA_FUNCTION_NULL = 0,
        DELTA_N_BITS,
        DELTA_BOOL_OR,      // Logic OR - good for exceptions
        DELTA_BOOL_XOR,     // Logic XOR - good to check if bits were changed
        DELTA_GET_PREVIOUS, // Preserve previous value
        DELTA_GET_LAST,     // Preserve last value
        DELTA_NS_TIME,      // Delta for nanosecond timestamps (GPU timestamp wraps at 32 bits but was value multiplied by 80)
        DELTA_FUNCTION_LAST_1_0
    } TDeltaFunctionType;

    //////////////////////////////////////////////////////////////////////////////////
    // Delta function:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SDeltaFunction_1_0
    {
        TDeltaFunctionType FunctionType;
        union
        {
            uint32_t BitsCount; // Used for DELTA_N_BITS to specify bits count
        };
    } TDeltaFunction_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Equation element types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EEquationElementType
    {
        EQUATION_ELEM_OPERATION,                 // See TEquationOperation enumeration
        EQUATION_ELEM_RD_BITFIELD,               //
        EQUATION_ELEM_RD_UINT8,                  //
        EQUATION_ELEM_RD_UINT16,                 //
        EQUATION_ELEM_RD_UINT32,                 //
        EQUATION_ELEM_RD_UINT64,                 //
        EQUATION_ELEM_RD_FLOAT,                  //
        EQUATION_ELEM_RD_40BIT_CNTR,             // Assemble 40 bit counter that is in two locations, result in unsigned integer 64b
        EQUATION_ELEM_IMM_UINT64,                //
        EQUATION_ELEM_IMM_FLOAT,                 //
        EQUATION_ELEM_SELF_COUNTER_VALUE,        // Defined by $Self token, the UINT64 result of DeltaFunction for IO or QueryReadEquation
        EQUATION_ELEM_GLOBAL_SYMBOL,             // Defined by $"SymbolName", available in MetricsDevice SymbolTable
        EQUATION_ELEM_LOCAL_COUNTER_SYMBOL,      // Defined by $"SymbolName", refers to counter delta value in the local set
        EQUATION_ELEM_OTHER_SET_COUNTER_SYMBOL,  // Defined by concatenated string of $"setSymbolName/SymbolName", refers to counter
                                                 // Delta value in the other set
        EQUATION_ELEM_LOCAL_METRIC_SYMBOL,       // Defined by $$"SymbolName", refers to metric normalized value in the local set
        EQUATION_ELEM_OTHER_SET_METRIC_SYMBOL,   // Defined by concatenated string of $$"setSymbolName/SymbolName", refers to metric
                                                 // Normalized value in the other set
        EQUATION_ELEM_INFORMATION_SYMBOL,        // Defined by i$"SymbolName", refers to information value type only
        EQUATION_ELEM_STD_NORM_GPU_DURATION,     // Action is $Self $GpuCoreClocks FDIV 100 FMUL
        EQUATION_ELEM_STD_NORM_EU_AGGR_DURATION, // Action is $Self $GpuCoreClocks $EuCoresTotalCount UMUL FDIV 100 FMUL
        EQUATION_ELEM_MASK,                      //
        EQUATION_ELEM_PREV_METRIC_SYMBOL,        // Defined by prev$$"SymbolName", refers to previous metric value in the local set
        EQUATION_ELEM_LAST_1_0

    } TEquationElementType;

    //////////////////////////////////////////////////////////////////////////////////
    // Equation operations:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EEquationOperation
    {
        EQUATION_OPER_RSHIFT, // 64b unsigned integer right shift
        EQUATION_OPER_LSHIFT, // 64b unsigned integer left shift
        EQUATION_OPER_AND,    // Bitwise AND of two unsigned integers, 64b each
        EQUATION_OPER_OR,     // Bitwise OR of two unsigned integers, 64b each
        EQUATION_OPER_XOR,    // Bitwise XOR of two unsigned integers, 64b each
        EQUATION_OPER_XNOR,   // Bitwise XNOR of two unsigned integers, 64b each
        EQUATION_OPER_AND_L,  // Logical AND (C-like "&&") of two unsigned integers, 64b each, result is true(1) if both values are true(greater than 0)
        EQUATION_OPER_EQUALS, // Equality (C-like "==") of two unsigned integers, 64b each, result is true(1) or false(0)
        EQUATION_OPER_UADD,   // Unsigned integer add, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_USUB,   // Unsigned integer subtract, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_UMUL,   // Unsigned integer mul, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_UDIV,   // Unsigned integer div, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_FADD,   // Floating point add, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_FSUB,   // Floating point subtract, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_FMUL,   // Floating point multiply, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_FDIV,   // Floating point divide, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_UGT,    // 64b unsigned integers comparison of is greater than, result is bool true(1) or false(0)
        EQUATION_OPER_ULT,    // 64b unsigned integers comparison of is less than, result is bool true(1) or false(0)
        EQUATION_OPER_UGTE,   // 64b unsigned integers comparison of is greater than or equal, result is bool true(1) or false(0)
        EQUATION_OPER_ULTE,   // 64b unsigned integers comparison of is less than or equal, result is bool true(1) or false(0)
        EQUATION_OPER_FGT,    // 32b floating point numbers comparison of is greater than, result is bool true(1) or false(0)
        EQUATION_OPER_FLT,    // 32b floating point numbers comparison of is less than, result is bool true(1) or false(0)
        EQUATION_OPER_FGTE,   // 32b floating point numbers comparison of is greater than or equal, result is bool true(1) or false(0)
        EQUATION_OPER_FLTE,   // 32b floating point numbers comparison of is less than or equal, result is bool true(1) or false(0)
        EQUATION_OPER_UMIN,   // Unsigned integer MIN function, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_UMAX,   // Unsigned integer MAX function, arguments are casted to be 64b unsigned integers, result is unsigned integer 64b
        EQUATION_OPER_FMIN,   // Floating point MIN function, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_FMAX,   // Floating point MAX function, arguments are casted to be 32b floating points, result is a 32b float
        EQUATION_OPER_LAST_1_0
    } TEquationOperation;

    //////////////////////////////////////////////////////////////////////////////////
    // Metric prototype option descriptor types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EOptionDescriptorType
    {
        OPTION_DESCRIPTOR_TYPE_DISAGGREGATION            = 0, // Metric is disaggregated by: slices, xe cores, l3 banks etc
        OPTION_DESCRIPTOR_TYPE_LATENCY                   = 1, // Latency measurements
        OPTION_DESCRIPTOR_TYPE_NORMALIZATION_UTILIZATION = 2, // Produces normalization equation: "raw_metric * 100% / cycles / instance_count", metric unit is updated to percent
        OPTION_DESCRIPTOR_TYPE_NORMALIZATION_AVERAGE     = 3, // Produces normalization equation: "raw_metric / instance_count", metric unit is not changed
        OPTION_DESCRIPTOR_TYPE_NORMALIZATION_RATE        = 4, // Produces normalization equation: "raw_metric / timestamp", metric unit is updated to per second
        OPTION_DESCRIPTOR_TYPE_NORMALIZATION_BYTE        = 5, // Produces normalization equation: "raw_metric * N bytes", metric unit is updated to bytes
        OPTION_DESCRIPTOR_TYPE_LAST
    } TOptionDescriptorType;

    //////////////////////////////////////////////////////////////////////////////////
    // Disaggregation modes:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EDisaggregationMode
    {
        DISAGGREGATION_MODE_NONE       = 0,
        DISAGGREGATION_MODE_XECORE     = 1,
        DISAGGREGATION_MODE_L3BANK     = 2,
        DISAGGREGATION_MODE_SLICE      = 3,
        DISAGGREGATION_MODE_SQIDI      = 4,
        DISAGGREGATION_MODE_L3NODE     = 5,
        DISAGGREGATION_MODE_COPYENGINE = 6,
        DISAGGREGATION_MODE_CXL        = 7,
        DISAGGREGATION_MODE_LAST
    } TDisaggregationMode;

    //////////////////////////////////////////////////////////////////////////////////
    // Query modes:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EQueryMode
    {
        QUERY_MODE_NONE            = 0,
        QUERY_MODE_RENDER          = 1,
        QUERY_MODE_COMPUTE         = 2,
        QUERY_MODE_GLOBAL          = 3,
        QUERY_MODE_GLOBAL_EXTENDED = 4,
        QUERY_MODE_LAST
    } TQueryMode;

    //////////////////////////////////////////////////////////////////////////////////
    // Query mode masks:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EQueryModeMask
    {
        QUERY_MODE_MASK_NONE            = 0x00000000,
        QUERY_MODE_MASK_RENDER          = 0x00000001,
        QUERY_MODE_MASK_COMPUTE         = 0x00000002,
        QUERY_MODE_MASK_GLOBAL          = 0x00000004,
        QUERY_MODE_MASK_GLOBAL_EXTENDED = 0x00000008,
        QUERY_MODE_MASK_ALL             = 0xFFFFFFFF
    } TQueryModeMask;

    //////////////////////////////////////////////////////////////////////////////////
    // IO Stream states:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum EIoStreamState
    {
        IO_STREAM_STATE_DISABLED = 0,
        IO_STREAM_STATE_ENABLED,
    } TIoStreamState;

    //////////////////////////////////////////////////////////////////////////////////
    // Counters modes in flexible metric sets:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum ECountersMode
    {
        COUNTERS_MODE_NORMAL = 0, // 64 Performance Event Counters (64-bit)
        COUNTERS_MODE_DENSE,      // 128 Performance Event Counters (32-bit)
    } TCountersMode;

    //////////////////////////////////////////////////////////////////////////////////
    // Read params:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SReadParams_1_0
    {
        uint32_t ByteOffset;
        uint32_t BitOffset;
        uint32_t BitsCount;
        uint32_t ByteOffsetExt;
    } TReadParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Equation element:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SEquationElement_1_0
    {
        TEquationElementType Type;
        union
        {
            uint64_t           ImmediateUInt64;
            float              ImmediateFloat;
            TByteArray_1_0     Mask;
            TEquationOperation Operation;
            TReadParams_1_0    ReadParams;
        };
        char* SymbolName;
    } TEquationElement_1_0;

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IEquation_1_0
    //
    // Description:
    //   Abstract interface for the equation object.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IEquation_1_0
    {
    public:
        virtual ~IEquation_1_0();
        virtual uint32_t              GetEquationElementsCount( void );
        virtual TEquationElement_1_0* GetEquationElement( uint32_t index );
    };

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metric:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricParams_1_0
    {
        uint32_t           IdInSet;           // Position in set (may change after SetApiFiltering)
        uint32_t           GroupId;           // Specific metric group id
        const char*        SymbolName;        // Symbol name, used in equations
        const char*        ShortName;         // Consistent metric name, not changed platform to platform
        const char*        GroupName;         // VertexShader for example
        const char*        LongName;          // Hint about the metric shown to users
        const char*        DxToOglAlias;      // To replace DX pixels with OGL fragments
        uint32_t           UsageFlagsMask;    //
        uint32_t           ApiMask;           //
        TMetricResultType  ResultType;        //
        const char*        MetricResultUnits; //
        TMetricType        MetricType;        //
        uint64_t           LowWatermark;      // Low watermark for hotspot indication (USAGE_FLAG_INDICATE only)
        uint64_t           HighWatermark;     // High watermark for hotspot indication (USAGE_FLAG_INDICATE only)
        THwUnitType        HwUnitType;        //
        IEquation_1_0*     IoReadEquation;    // Read equation specification for IO stream (accessing raw values potentially spread in report in several locations)
        IEquation_1_0*     QueryReadEquation; // Read equation specification for query (accessing calculated delta values)
        TDeltaFunction_1_0 DeltaFunction;     //
        IEquation_1_0*     NormEquation;      // Normalization equation to get normalized value to bytes transfered or to a percentage of utilization
        IEquation_1_0*     MaxValueEquation;  // To calculate metrics max value as a function of other metrics and device parameters (e.g. 100 for percentage)
    } TMetricParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Extended global parameters of Metric:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricParams_1_13 : public SMetricParams_1_0
    {
        uint32_t QueryModeMask; // Mask values specified in TQueryModeMask
    } TMetricParams_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Valid value:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SValidValue_1_13
    {
        TValueType ValueType;
        union
        {
            uint32_t ValueUInt32;
            uint64_t ValueUInt64;
            struct
            {
                uint32_t Min;
                uint32_t Max;
            } ValueUInt32Range;
            struct
            {
                uint64_t Min;
                uint64_t Max;
            } ValueUInt64Range;
        };
    } TValidValue_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metric Prototype:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricPrototypeParams_1_13
    {
        const char*         SymbolName;            // Symbol name, used in equations
        const char*         ShortName;             // Consistent metric name, not changed platform to platform
        const char*         GroupName;             // VertexShader for example
        const char*         LongName;              // Hint about the metric shown to users
        const char*         DxToOglAlias;          // To replace DX pixels with OGL fragments
        uint32_t            UsageFlagsMask;        //
        uint32_t            ApiMask;               //
        TMetricResultType   ResultType;            //
        const char*         MetricResultUnits;     //
        TMetricType         MetricType;            //
        THwUnitType         HwUnitType;            //
        uint32_t            OptionDescriptorCount; //
        TDisaggregationMode DisaggregationMode;    //
        uint32_t            QueryModeMask;         // Mask values specified in TQueryModeMask
    } TMetricPrototypeParams_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Metric Prototype Option Descriptor:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SMetricPrototypeOptionDescriptor_1_13
    {
        TOptionDescriptorType Type;
        const char*           SymbolName;
        TValidValue_1_13*     ValidValues;
        uint32_t              ValidValueCount;
    } TMetricPrototypeOptionDescriptor_1_13;

    //////////////////////////////////////////////////////////////////////////////////
    // Global parameters of Information:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SInformationParams_1_0
    {
        uint32_t           IdInSet;           // Position in set (may change after SetApiFiltering)
        const char*        SymbolName;        // Symbol name, used in equations
        const char*        ShortName;         // Consistent name, not changed platform to platform
        const char*        GroupName;         // Some more global context of the information
        const char*        LongName;          // Hint about the information shown to users
        uint32_t           ApiMask;           //
        TInformationType   InfoType;          //
        const char*        InfoUnits;         //
        IEquation_1_0*     IoReadEquation;    // Read equation specification for IO stream (accessing raw values potentially spread in report in several locations)
        IEquation_1_0*     QueryReadEquation; // Read equation specification for query (accessing calculated delta values)
        TDeltaFunction_1_0 OverflowFunction;  //
    } TInformationParams_1_0;

    //////////////////////////////////////////////////////////////////////////////////
    // Calculation context types:
    //////////////////////////////////////////////////////////////////////////////////
    typedef enum ECalculationContextType
    {
        CALCULATION_CONTEXT_TYPE_IO_STREAM,
        CALCULATION_CONTEXT_TYPE_QUERY,
        CALCULATION_CONTEXT_TYPE_LAST
    } TCalculationContextType;

    //////////////////////////////////////////////////////////////////////////////////
    // Time windows:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct STimeWindow_1_16
    {
        uint64_t Start; // Start timestamp in nanoseconds
        uint64_t End;   // End timestamp in nanoseconds
    } TTimeWindow_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Calculation context descriptor for IO Stream:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SCalculationContextIoStreamDescriptor_1_16
    {
        TTimeWindow_1_16* TimeWindows;             // (Optional) Array of time windows, each containing a start time and an end time in nanoseconds
        uint32_t          TimeWindowCount;         // Number of time windows
        uint64_t          NsTimeAggregationWindow; // Aggregation window size in nanoseconds; if 0, no aggregation is performed (applies to time windows if provided)
    } TCalculationContextIoStreamDescriptor_1_16;

    //////////////////////////////////////////////////////////////////////////////////
    // Calculation context descriptor for Query:
    //////////////////////////////////////////////////////////////////////////////////
    typedef struct SCalculationContextQueryDescriptor_1_16
    {
    } TCalculationContextQueryDescriptor_1_16;

    ///////////////////////////////////////////////////////////////////////////////
    // Calculation context descriptor:
    ///////////////////////////////////////////////////////////////////////////////
    typedef struct SCalculationContextDescriptor_1_16
    {
        TCalculationContextType Type; // Type of the descriptor (IO Stream or Query)

        uint32_t          DataSetCount; // Number of metric sets and raw data sets to aggregate (DataSetCount == 1: time aggregation only, DataSetCount > 1: cross-subdevice aggregation)
        IMetricSet_1_16** MetricSets;   // Array of metric sets from multiple sub devices ([0..DataSetCount]), all metric sets must be identical (e.g. cannot mix ComputeBasic with VectorEngineProfile)
        int64_t*          TimeOffsets;  // (Optional) Array of time offsets if timestamps are not synchronized across sub devices ([0..DataSetCount])

        union
        {
            TCalculationContextIoStreamDescriptor_1_16 IoStreamDescriptor; // Descriptor for IO Stream
            TCalculationContextQueryDescriptor_1_16    QueryDescriptor;    // Descriptor for Query
        };
    } TCalculationContextDescriptor_1_16;

    ///////////////////////////////////////////////////////////////////////////////
    // Global parameters of calculation context:
    ///////////////////////////////////////////////////////////////////////////////
    typedef struct SCalculationContextParams_1_16
    {
        uint32_t MetricsCount;
        uint32_t InformationCount;
    } TCalculationContextParams_1_16;

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IInformation_1_0
    //
    // Descriptions:
    //   Abstract interface for the measurement information parameter.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IInformation_1_0
    {
    public:
        virtual ~IInformation_1_0();
        virtual TInformationParams_1_0* GetParams();
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetric_1_0
    //
    // Description:
    //   Abstract interface for the metric that is sampled.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetric_1_0
    {
    public:
        virtual ~IMetric_1_0();
        virtual TMetricParams_1_0* GetParams();
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetric_1_13
    //
    // Description:
    //   Updated 1.0 version to use with 1.13 interface version.
    //   Introduces new metric params.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetric_1_13 : public IMetric_1_0
    {
    public:
        virtual ~IMetric_1_13();
        virtual TMetricParams_1_13* GetParams();
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricPrototype_1_13
    //
    // Description:
    //   Abstract interface for the metric prototype that is sampled.
    //
    // New:
    // - GetParams:                 To get this metric prototype params
    // - Clone:                     To clone the metric prototype
    // - GetOptionDescriptor:       To get option descriptor
    // - SetOption:                 To set option
    // - ChangeNames:               To change symbol, short, long or unit names
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricPrototype_1_13
    {
    public:
        virtual ~IMetricPrototype_1_13();
        virtual const TMetricPrototypeParams_1_13*           GetParams( void ) const;
        virtual IMetricPrototype_1_13*                       Clone( void );
        virtual const TMetricPrototypeOptionDescriptor_1_13* GetOptionDescriptor( uint32_t index ) const;
        virtual TCompletionCode                              SetOption( const TOptionDescriptorType optionType, const TTypedValue_1_0* typedValue );
        virtual TCompletionCode                              ChangeNames( const char* symbolName, const char* shortName, const char* longName, const char* resultUnits );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricEnumerator_1_13
    //
    // Description:
    //   Abstract interface for the metric enumerator.
    //
    // New:
    // - GetMetricPrototypeCount:     To get the number of available metric prototypes
    // - GetMetricPrototype:          To get available metric prototype at a given index
    // - GetMetricPrototypes:         To get a given number of available metric prototypes
    // - RemoveClonedMetricPrototype: To remove cloned metric prototype
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricEnumerator_1_13
    {
    public:
        virtual ~IMetricEnumerator_1_13();
        virtual uint32_t               GetMetricPrototypeCount( void );
        virtual IMetricPrototype_1_13* GetMetricPrototype( const uint32_t index );
        virtual TCompletionCode        GetMetricPrototypes( const uint32_t index, uint32_t* count, IMetricPrototype_1_13** metrics );
        virtual TCompletionCode        RemoveClonedMetricPrototype( IMetricPrototype_1_13* clonedPrototype );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_0
    //
    // Description:
    //   Abstract interface for the metric sets mapping to different HW
    //   configuration that should be used exclusively to each other
    //   metric set in the concurrent group.
    //
    // New:
    // - GetParams:
    // - GetMetric:                         To get particular metric
    // - GetInformation:                    To get particular information about measurement
    // - GetComplementaryMetricSet:         Below proposal to address multi-passes at the set level
    // - Activate:                          To enable this configuration before query instance is created
    // - Deactivate:                        To disable this configuration after query instance is created
    // - AddCustomMetric:                   To add an additional custom metric to this set
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_0
    {
    public:
        virtual ~IMetricSet_1_0();

        virtual TMetricSetParams_1_0* GetParams( void );
        virtual IMetric_1_0*          GetMetric( uint32_t index );
        virtual IInformation_1_0*     GetInformation( uint32_t index );
        virtual IMetricSet_1_0*       GetComplementaryMetricSet( uint32_t index );
        virtual TCompletionCode       Activate( void );
        virtual TCompletionCode       Deactivate( void );
        virtual IMetric_1_0*          AddCustomMetric(
                     const char*       symbolName,
                     const char*       shortName,
                     const char*       groupName,
                     const char*       longName,
                     const char*       dxToOglAlias,
                     uint32_t          usageFlagsMask,
                     uint32_t          apiMask,
                     TMetricResultType resultType,
                     const char*       resultUnits,
                     TMetricType       metricType,
                     int64_t           loWatermark,
                     int64_t           hiWatermark,
                     THwUnitType       hwType,
                     const char*       ioReadEquation,
                     const char*       deltaFunction,
                     const char*       queryReadEquation,
                     const char*       normalizationEquation,
                     const char*       maxValueEquation,
                     const char*       signalName );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_1
    //
    // Description:
    //   Updated 1.0 version to use with 1.1 interface version.
    //   Introduces an ability to calculate metrics from raw data.
    //
    // New:
    // - SetApiFiltering:                       To filter available metrics/information for the given API. Use TMetricApiType to build the mask.
    // - CalculateMetrics:                      To calculate normalized metrics/information from the raw data.
    // - CalculateIoMeasurementInformation:     To calculate additional information for stream measurements.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_1 : public IMetricSet_1_0
    {
    public:
        virtual ~IMetricSet_1_1();
        virtual TCompletionCode SetApiFiltering( uint32_t apiMask );
        virtual TCompletionCode CalculateMetrics( const uint8_t* rawData, uint32_t rawDataSize, TTypedValue_1_0* out, uint32_t outSize, uint32_t* outReportCount, bool enableContextFiltering );
        virtual TCompletionCode CalculateIoMeasurementInformation( TTypedValue_1_0* out, uint32_t outSize );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_4
    //
    // Description:
    //   Updated 1.1 version to use with 1.4 interface version.
    //   Extends set params with gtType information.
    //
    // Updates:
    // - GetParams
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_4 : public IMetricSet_1_1
    {
    public:
        virtual ~IMetricSet_1_4();
        virtual TMetricSetParams_1_4* GetParams( void );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_5
    //
    // Description:
    //   Updated 1.4 version to use with 1.5 interface version.
    //   Adds an ability to calculate MaxValueEquations (maximal value) for each metric.
    //   Param 'enableContextFiltering' becomes deprecated.
    //
    // Updates:
    // - GetComplementaryMetricSet:         Update to 1.5 interface
    // - CalculateMetrics:                  CalculateMetrics extended with max values calculation.
    //                                      optional param 'outMaxValues' should have a memory
    //                                      for at least 'MetricCount * RawReportCount' values, can be nullptr.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_5 : public IMetricSet_1_4
    {
    public:
        using IMetricSet_1_1::CalculateMetrics; // To avoid hiding by 1.5 interface function
        virtual IMetricSet_1_5* GetComplementaryMetricSet( uint32_t index );
        virtual TCompletionCode CalculateMetrics(
            const uint8_t*   rawData,
            uint32_t         rawDataSize,
            TTypedValue_1_0* out,
            uint32_t         outSize,
            uint32_t*        outReportCount,
            TTypedValue_1_0* outMaxValues,
            uint32_t         outMaxValuesSize );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_11
    //
    // Description:
    //   Updated 1.5 version to use with 1.11 interface version.
    //   Extends set params with AvailabilityEquation information.
    //
    // Updates:
    // - GetParams
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_11 : public IMetricSet_1_5
    {
    public:
        virtual ~IMetricSet_1_11();
        virtual TMetricSetParams_1_11* GetParams( void );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_13
    //
    // Description:
    //   Updated 1.11 version to use with 1.13 interface version.
    //   Adds an ability to create and delete flexible metric sets.
    //
    // New:
    // - Open:              Prepare metric set for adding metrics
    // - AddMetric:         To add a given metric to flexible metric set
    // - RemoveMetric:      To remove a given metric from flexible metric set
    // - Finalize:          To make metric set usable after adding metrics
    //
    // Updates:
    // - GetMetric          Update to 1.13 interface
    // - AddCustomMetric    Update to 1.13 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_13 : public IMetricSet_1_11
    {
    public:
        virtual ~IMetricSet_1_13();

        // New.
        virtual TCompletionCode Open();
        virtual TCompletionCode AddMetric( IMetricPrototype_1_13* metricPrototype );
        virtual TCompletionCode RemoveMetric( IMetricPrototype_1_13* metricPrototype );
        virtual TCompletionCode Finalize();

        // Updates.
        virtual IMetric_1_13* GetMetric( uint32_t index );

        using IMetricSet_1_0::AddCustomMetric; // To avoid hiding by 1.13 interface function
        virtual IMetric_1_13* AddCustomMetric(
            const char*       symbolName,
            const char*       shortName,
            const char*       groupName,
            const char*       longName,
            const char*       dxToOglAlias,
            uint32_t          usageFlagsMask,
            uint32_t          apiMask,
            TMetricResultType resultType,
            const char*       resultUnits,
            TMetricType       metricType,
            int64_t           loWatermark,
            int64_t           hiWatermark,
            THwUnitType       hwType,
            const char*       ioReadEquation,
            const char*       deltaFunction,
            const char*       queryReadEquation,
            const char*       normalizationEquation,
            const char*       maxValueEquation,
            const char*       signalName,
            uint32_t          queryModeMask );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricSet_1_16
    //
    // Description:
    //   Updated 1.13 version to use with 1.16 interface version.
    //   Adds an ability to calculate async metrics from raw data.
    //
    // New:
    // - CalculateAsyncMetrics: To calculate async metrics from raw data.
    //
    // Updates:
    // - GetParams          Update to 1.16 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricSet_1_16 : public IMetricSet_1_13
    {
    public:
        virtual ~IMetricSet_1_16();

        // New.
        virtual TCompletionCode CalculateAsyncMetrics( const uint8_t* rawData, uint32_t rawDataSize, TTypedValue_1_0* out, uint32_t outSize, uint32_t* outReportCount, TTypedValue_1_0* outMaxValues, uint32_t outMaxValuesSize );

        // Updates.
        virtual TMetricSetParams_1_16* GetParams( void );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_0
    //
    // Description:
    //   Abstract interface for the metrics groups that can be collected
    //   concurrently to another group.
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_0
    {
    public:
        virtual ~IConcurrentGroup_1_0();
        virtual TConcurrentGroupParams_1_0* GetParams( void );
        virtual IMetricSet_1_0*             GetMetricSet( uint32_t index );
        virtual TCompletionCode             OpenIoStream( IMetricSet_1_0* metricSet, uint32_t processId, uint32_t* nsTimerPeriod, uint32_t* oaBufferSize );
        virtual TCompletionCode             ReadIoStream( uint32_t* reportsCount, char* reportData, uint32_t readFlags );
        virtual TCompletionCode             CloseIoStream( void );
        virtual TCompletionCode             WaitForReports( uint32_t milliseconds );
        virtual IInformation_1_0*           GetIoMeasurementInformation( uint32_t index );
        virtual IInformation_1_0*           GetIoGpuContextInformation( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_1
    //
    // Description:
    //   Updated 1.0 version to use with 1.1 interface version.
    //
    // Updates:
    // - GetMetricSet:                  Update to 1.1 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_1 : public IConcurrentGroup_1_0
    {
    public:
        virtual IMetricSet_1_1* GetMetricSet( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_3
    //
    // Description:
    //   Updated 1.1 version to use with 1.3 interface version.Introduces setting Stream Sampling Type.
    //
    // New:
    // - SetIoStreamSamplingType:       To set sampling type during IoStream measurements
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_3 : public IConcurrentGroup_1_1
    {
    public:
        virtual TCompletionCode SetIoStreamSamplingType( TSamplingType type );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_5
    //
    // Description:
    //   Updated 1.3 version to use with 1.5 interface version.
    //
    // Updates:
    // - GetMetricSet:                  Update to 1.5 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_5 : public IConcurrentGroup_1_3
    {
    public:
        virtual IMetricSet_1_5* GetMetricSet( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_11
    //
    // Description:
    //   Updated 1.5 version to use with 1.11 interface version.
    //
    // Updates:
    // - GetMetricSet:                  Update to 1.11 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_11 : public IConcurrentGroup_1_5
    {
    public:
        virtual IMetricSet_1_11* GetMetricSet( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_13
    //
    // Description:
    //   Updated 1.11 version to use with 1.13 interface version.
    //
    // Updates:
    // - GetParams                      Update to 1.13 interface
    // - GetMetricSet:                  Update to 1.13 interface
    //
    // New:
    // - GetMetricEnumerator:           To get metric enumerator
    // - GetMetricEnumeratorFromFile:   To get metric enumerator from file
    // - AddMetricSet:                  To create flexible metric set
    // - RemoveMetricSet:               To delete flexible metric set
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_13 : public IConcurrentGroup_1_11
    {
    public:
        // Updates.
        virtual TConcurrentGroupParams_1_13* GetParams( void );
        virtual IMetricSet_1_13*             GetMetricSet( uint32_t index );

        // New.
        virtual IMetricEnumerator_1_13* GetMetricEnumerator( void );
        virtual IMetricEnumerator_1_13* GetMetricEnumeratorFromFile( const char* fileName );
        virtual IMetricSet_1_13*        AddMetricSet( const char* symbolName, const char* shortName );
        virtual TCompletionCode         RemoveMetricSet( IMetricSet_1_13* metricSet );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_15
    //
    // Description:
    //   Updated 1.13 version to use with 1.15 interface version.
    //
    // Updates:
    // - OpenIoStream:                  Update to 1.15 interface
    //
    // New:
    // - ChangeIoStreamState:           To change IO stream state
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_15 : public IConcurrentGroup_1_13
    {
    public:
        // Updates.
        using IConcurrentGroup_1_13::OpenIoStream; // To avoid hiding by 1.15 interface function

        virtual TCompletionCode OpenIoStream( IMetricSet_1_13* metricSet, uint32_t processId, uint32_t* nsTimerPeriod, uint32_t* oaBufferSize, TIoStreamState state );
        virtual TCompletionCode ChangeIoStreamState( TIoStreamState state, uint32_t* nsTimerPeriod );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IConcurrentGroup_1_16
    //
    // Description:
    //   Updated 1.15 version to use with 1.16 interface version.
    //
    // Updates:
    // - GetMetricSet:                  Update to 1.16 interface
    // - AddMetricSet:                  Update to 1.16 interface
    // - RemoveMetricSet:               Update to 1.16 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IConcurrentGroup_1_16 : public IConcurrentGroup_1_15
    {
    public:
        // Updates.
        using IConcurrentGroup_1_15::AddMetricSet;    // To avoid hiding by 1.16 interface function
        using IConcurrentGroup_1_15::RemoveMetricSet; // To avoid hiding by 1.16 interface function

        virtual IMetricSet_1_16* GetMetricSet( uint32_t index );
        virtual IMetricSet_1_16* AddMetricSet( const char* symbolName, const char* shortName, TCountersMode mode );
        virtual TCompletionCode  RemoveMetricSet( IMetricSet_1_16* metricSet );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IOverride_1_2
    //
    // Description:
    //   Abstract interface for Metrics Device overrides.
    //
    // New:
    // - GetParams:                     To get this Override params
    // - SetOverride:                   To enable/disable this Override
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IOverride_1_2
    {
    public:
        virtual ~IOverride_1_2();
        virtual TOverrideParams_1_2* GetParams( void );
        virtual TCompletionCode      SetOverride( TSetOverrideParams_1_2* params, uint32_t paramsSize );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_0
    //
    // Description:
    //   Abstract interface for the GPU metrics root object.
    //
    // New:
    // - GetParams:                     To get MetricsDevice params
    // - GetConcurrentGroup:            Child objects are of IConcurrentGroup
    // - GetGlobalSymbol:               To get GlobalSymbol at the given index
    // - GetGlobalSymbolValueByName:    To get GlobalSymbol with the given name
    // - GetLastError:                  To get last error from TCompletionCode enum
    // - GetGpuCpuTimestamps:           To get both GPU and CPU timestamp at the same time
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_0
    {
    public:
        virtual ~IMetricsDevice_1_0();
        virtual TMetricsDeviceParams_1_0* GetParams( void );
        virtual IConcurrentGroup_1_0*     GetConcurrentGroup( uint32_t index );
        virtual TGlobalSymbol_1_0*        GetGlobalSymbol( uint32_t index );
        virtual TTypedValue_1_0*          GetGlobalSymbolValueByName( const char* name );
        virtual TCompletionCode           GetLastError( void );
        virtual TCompletionCode           GetGpuCpuTimestamps( uint64_t* gpuTimestampNs, uint64_t* cpuTimestampNs, uint32_t* cpuId );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_1
    //
    // Description:
    //   Updated 1.0 version to use with 1.1 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.1 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_1 : public IMetricsDevice_1_0
    {
    public:
        virtual IConcurrentGroup_1_1* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_2
    //
    // Description:
    //   Updated 1.1 version to use with 1.2 interface version.
    //   Introduces an interface for getting overrides.
    //
    // Updates:
    // - GetParams:                     Update to 1.2 interface
    //
    // New:
    // - GetOverride:                   To get override at the given index
    // - GetOverrideByName:             To get override with the given name
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_2 : public IMetricsDevice_1_1
    {
    public:
        // Updates.
        virtual TMetricsDeviceParams_1_2* GetParams( void );

        // New.
        virtual IOverride_1_2* GetOverride( uint32_t index );
        virtual IOverride_1_2* GetOverrideByName( const char* symbolName );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_5
    //
    // Description:
    //   Updated 1.2 version to use with 1.5 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.5 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_5 : public IMetricsDevice_1_2
    {
    public:
        virtual IConcurrentGroup_1_5* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_10
    //
    // Description:
    //   Updated 1.5 version to use with 1.10 interface version.
    //
    // Updates:
    // - GetGpuCpuTimestamps:            Update to 1.10 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_10 : public IMetricsDevice_1_5
    {
    public:
        using IMetricsDevice_1_0::GetGpuCpuTimestamps; // To avoid hiding by 1.10 interface function
        virtual TCompletionCode GetGpuCpuTimestamps( uint64_t* gpuTimestampNs, uint64_t* cpuTimestampNs, uint32_t* cpuId, uint64_t* correlationIndicatorNs );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_11
    //
    // Description:
    //   Updated 1.10 version to use with 1.11 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.11 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_11 : public IMetricsDevice_1_10
    {
    public:
        virtual IConcurrentGroup_1_11* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_13
    //
    // Description:
    //   Updated 1.11 version to use with 1.13 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.13 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_13 : public IMetricsDevice_1_11
    {
    public:
        virtual IConcurrentGroup_1_13* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_15
    //
    // Description:
    //   Updated 1.13 version to use with 1.15 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.15 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_15 : public IMetricsDevice_1_13
    {
    public:
        virtual IConcurrentGroup_1_15* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IMetricsDevice_1_16
    //
    // Description:
    //   Updated 1.15 version to use with 1.16 interface version.
    //
    // Updates:
    // - GetConcurrentGroup:            Update to 1.16 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IMetricsDevice_1_16 : public IMetricsDevice_1_15
    {
    public:
        virtual IConcurrentGroup_1_16* GetConcurrentGroup( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   Calculation Context interface.
    //
    // New:
    // - GetParams:                     To get CalculationContext params
    // - GetMetric:                     To get particular metric
    // - GetInformation:                To get particular information
    // - AggregateData:                 To aggregate data
    // - CalculateMetrics:              To calculate normalized metrics/information from the raw data
    // - CalculateSingleWindowMetrics:  To aggregate data for only a single aggregation window and to calculate normalized metrics/information from aggregated raw data
    //
    ///////////////////////////////////////////////////////////////////////////////
    class ICalculationContext_1_16
    {
    public:
        virtual ~ICalculationContext_1_16();

        virtual const TCalculationContextParams_1_16* GetParams( void ) const;
        virtual IMetric_1_13*                         GetMetric( uint32_t index );
        virtual IInformation_1_0*                     GetInformation( uint32_t index );

        virtual TCompletionCode AggregateData( const uint8_t** rawData, const uint32_t* rawDataSizes, uint8_t* outAggregatedRawData, uint32_t* outAggregatedRawDataSize, bool lastDataPortion );
        virtual TCompletionCode CalculateMetrics( const uint8_t* rawData, uint32_t rawDataSize, TTypedValue_1_0* out, uint32_t outSize, uint32_t* outReportCount, TTypedValue_1_0* outMaxValues, uint32_t outMaxValuesSize );
        virtual TCompletionCode CalculateSingleWindowMetrics( const uint8_t** rawData, const uint32_t* rawDataSizes, uint32_t* outProcessedRawDataCount, TTypedValue_1_0* out, uint32_t outSize, TTypedValue_1_0* outMaxValues, uint32_t outMaxValuesSize, bool lastDataPortion );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_6
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // New:
    // - GetParams:                     To get this adapter params
    // - Reset:                         To reset this adapter state
    // - OpenMetricsDevice
    // - OpenMetricsDeviceFromFile
    // - CloseMetricsDevice
    // - SaveMetricsDeviceToFile
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_6
    {
    public:
        virtual ~IAdapter_1_6();
        virtual const TAdapterParams_1_6* GetParams( void ) const;
        virtual TCompletionCode           Reset();
        virtual TCompletionCode           OpenMetricsDevice( IMetricsDevice_1_5** metricsDevice );
        virtual TCompletionCode           OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_5** metricsDevice );
        virtual TCompletionCode           CloseMetricsDevice( IMetricsDevice_1_5* metricsDevice );
        virtual TCompletionCode           SaveMetricsDeviceToFile( const char* fileName, void* saveParams, IMetricsDevice_1_5* metricsDevice );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_8
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - GetParams:                     Update to 1.8 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_8 : public IAdapter_1_6
    {
    public:
        virtual const TAdapterParams_1_8* GetParams( void ) const;
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_9
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - GetParams:                     Update to 1.9 interface
    //
    // New:
    // - GetSubDeviceParams:            To get sub device parameters
    // - GetEngineParams:               To get engine parameters
    // - OpenMetricsSubDevice:          To open metrics device on given sub device
    // - OpenMetricsSubDeviceFromFile:  To open metrics device from file on given sub device
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_9 : public IAdapter_1_8
    {
    public:
        // Updates.
        virtual const TAdapterParams_1_9* GetParams( void ) const;

        // New.
        virtual const TSubDeviceParams_1_9* GetSubDeviceParams( const uint32_t subDeviceIndex );
        virtual const TEngineParams_1_9*    GetEngineParams( const uint32_t subDeviceIndex, const uint32_t engineIndex );
        virtual TCompletionCode             OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_5** metricsDevice );
        virtual TCompletionCode             OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_5** metricsDevice );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_10
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - OpenMetricsDevice:              Update to 1.10 interface
    // - OpenMetricsDeviceFromFile:      Update to 1.10 interface
    // - OpenMetricsSubDevice:           Update to 1.10 interface
    // - OpenMetricsSubDeviceFromFile:   Update to 1.10 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_10 : public IAdapter_1_9
    {
    public:
        using IAdapter_1_6::OpenMetricsDevice;            // To avoid hiding by 1.10 interface function
        using IAdapter_1_6::OpenMetricsDeviceFromFile;    // To avoid hiding by 1.10 interface function
        using IAdapter_1_9::OpenMetricsSubDevice;         // To avoid hiding by 1.10 interface function
        using IAdapter_1_9::OpenMetricsSubDeviceFromFile; // To avoid hiding by 1.10 interface function

        virtual TCompletionCode OpenMetricsDevice( IMetricsDevice_1_10** metricsDevice );
        virtual TCompletionCode OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_10** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_10** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_10** metricsDevice );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_11
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - OpenMetricsDevice:              Update to 1.11 interface
    // - OpenMetricsDeviceFromFile:      Update to 1.11 interface
    // - OpenMetricsSubDevice:           Update to 1.11 interface
    // - OpenMetricsSubDeviceFromFile:   Update to 1.11 interface
    //
    // New:
    // - SaveMetricsDeviceToFile   To save metrics device with required minimal api version
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_11 : public IAdapter_1_10
    {
    public:
        // Updates.
        using IAdapter_1_10::OpenMetricsDevice;            // To avoid hiding by 1.11 interface function
        using IAdapter_1_10::OpenMetricsDeviceFromFile;    // To avoid hiding by 1.11 interface function
        using IAdapter_1_10::OpenMetricsSubDevice;         // To avoid hiding by 1.11 interface function
        using IAdapter_1_10::OpenMetricsSubDeviceFromFile; // To avoid hiding by 1.11 interface function

        virtual TCompletionCode OpenMetricsDevice( IMetricsDevice_1_11** metricsDevice );
        virtual TCompletionCode OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_11** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_11** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_11** metricsDevice );

        // New.
        using IAdapter_1_6::SaveMetricsDeviceToFile;
        virtual TCompletionCode SaveMetricsDeviceToFile( const char* fileName, void* saveParams, IMetricsDevice_1_11* metricsDevice, const uint32_t minMajorApiVersion, const uint32_t minMinorApiVersion );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_13
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - OpenMetricsDevice:             Update to 1.13 interface
    // - OpenMetricsDeviceFromFile:     Update to 1.13 interface
    // - OpenMetricsSubDevice:          Update to 1.13 interface
    // - OpenMetricsSubDeviceFromFile:  Update to 1.13 interface
    // - GetEngineParams:               Update to 1.13 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_13 : public IAdapter_1_11
    {
    public:
        // Updates.
        using IAdapter_1_11::OpenMetricsDevice;            // To avoid hiding by 1.13 interface function
        using IAdapter_1_11::OpenMetricsDeviceFromFile;    // To avoid hiding by 1.13 interface function
        using IAdapter_1_11::OpenMetricsSubDevice;         // To avoid hiding by 1.13 interface function
        using IAdapter_1_11::OpenMetricsSubDeviceFromFile; // To avoid hiding by 1.13 interface function

        virtual TCompletionCode OpenMetricsDevice( IMetricsDevice_1_13** metricsDevice );
        virtual TCompletionCode OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_13** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_13** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_13** metricsDevice );

        virtual const TEngineParams_1_13* GetEngineParams( const uint32_t subDeviceIndex, const uint32_t engineIndex );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_15
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - OpenMetricsDevice:             Update to 1.15 interface
    // - OpenMetricsDeviceFromFile:     Update to 1.15 interface
    // - OpenMetricsSubDevice:          Update to 1.15 interface
    // - OpenMetricsSubDeviceFromFile:  Update to 1.15 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_15 : public IAdapter_1_13
    {
    public:
        // Updates.
        using IAdapter_1_13::OpenMetricsDevice;            // To avoid hiding by 1.15 interface function
        using IAdapter_1_13::OpenMetricsDeviceFromFile;    // To avoid hiding by 1.15 interface function
        using IAdapter_1_13::OpenMetricsSubDevice;         // To avoid hiding by 1.15 interface function
        using IAdapter_1_13::OpenMetricsSubDeviceFromFile; // To avoid hiding by 1.15 interface function

        virtual TCompletionCode OpenMetricsDevice( IMetricsDevice_1_15** metricsDevice );
        virtual TCompletionCode OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_15** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_15** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_15** metricsDevice );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapter_1_16
    //
    // Description:
    //   Abstract interface for GPU adapter.
    //
    // Updates:
    // - OpenMetricsDevice:             Update to 1.16 interface
    // - OpenMetricsDeviceFromFile:     Update to 1.16 interface
    // - OpenMetricsSubDevice:          Update to 1.16 interface
    // - OpenMetricsSubDeviceFromFile:  Update to 1.16 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapter_1_16 : public IAdapter_1_15
    {
    public:
        // Updates.
        using IAdapter_1_15::OpenMetricsDevice;            // To avoid hiding by 1.16 interface function
        using IAdapter_1_15::OpenMetricsDeviceFromFile;    // To avoid hiding by 1.16 interface function
        using IAdapter_1_15::OpenMetricsSubDevice;         // To avoid hiding by 1.16 interface function
        using IAdapter_1_15::OpenMetricsSubDeviceFromFile; // To avoid hiding by 1.16 interface function

        virtual TCompletionCode OpenMetricsDevice( IMetricsDevice_1_16** metricsDevice );
        virtual TCompletionCode OpenMetricsDeviceFromFile( const char* fileName, void* openParams, IMetricsDevice_1_16** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDevice( const uint32_t subDeviceIndex, IMetricsDevice_1_16** metricsDevice );
        virtual TCompletionCode OpenMetricsSubDeviceFromFile( const uint32_t subDeviceIndex, const char* fileName, void* openParams, IMetricsDevice_1_16** metricsDevice );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_6
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // New:
    // - GetParams:                     To get this adapter group params
    // - GetAdapter:                    To enumerate available GPU adapters
    // - Close:                         To close this adapter group
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_6
    {
    public:
        virtual ~IAdapterGroup_1_6();
        virtual const TAdapterGroupParams_1_6* GetParams( void ) const;
        virtual IAdapter_1_6*                  GetAdapter( uint32_t index );
        virtual TCompletionCode                Close();
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_8
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                    Update to 1.8 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_8 : public IAdapterGroup_1_6
    {
    public:
        virtual IAdapter_1_8* GetAdapter( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_9
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                    Update to 1.9 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_9 : public IAdapterGroup_1_8
    {
    public:
        virtual IAdapter_1_9* GetAdapter( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_10
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                    Update to 1.10 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_10 : public IAdapterGroup_1_9
    {
    public:
        virtual IAdapter_1_10* GetAdapter( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_11
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                    Update to 1.11 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_11 : public IAdapterGroup_1_10
    {
    public:
        virtual IAdapter_1_11* GetAdapter( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_13
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                    Update to 1.13 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_13 : public IAdapterGroup_1_11
    {
    public:
        virtual IAdapter_1_13* GetAdapter( uint32_t index );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_14
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // New:
    // - OpenOfflineMetricsDeviceFromBuffer: To open offline metrics device from buffer
    // - CloseOfflineMetricsDevice:          To close offline metrics device
    // - SaveMetricsDeviceToBuffer:          To save metrics device with chosen sets to buffer
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_14 : public IAdapterGroup_1_13
    {
    public:
        virtual TCompletionCode OpenOfflineMetricsDeviceFromBuffer( uint8_t* buffer, uint32_t bufferSize, IMetricsDevice_1_13** metricsDevice );
        virtual TCompletionCode CloseOfflineMetricsDevice( IMetricsDevice_1_13* metricsDevice );
        virtual TCompletionCode SaveMetricsDeviceToBuffer( IMetricsDevice_1_13* metricsDevice, IMetricSet_1_13** metricSets, uint32_t metricSetCount, uint8_t* buffer, uint32_t* bufferSize, const uint32_t minMajorApiVersion, const uint32_t minMinorApiVersion );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_15
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                         Update to 1.15 interface
    // - OpenOfflineMetricsDeviceFromBuffer: Update to 1.15 interface
    // - CloseOfflineMetricsDevice:          Update to 1.15 interface
    // - SaveMetricsDeviceToBuffer:          Update to 1.15 interface
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_15 : public IAdapterGroup_1_14
    {
    public:
        // Updates.
        using IAdapterGroup_1_14::OpenOfflineMetricsDeviceFromBuffer; // To avoid hiding by 1.15 interface function
        using IAdapterGroup_1_14::CloseOfflineMetricsDevice;          // To avoid hiding by 1.15 interface function
        using IAdapterGroup_1_14::SaveMetricsDeviceToBuffer;          // To avoid hiding by 1.15 interface function

        virtual IAdapter_1_15* GetAdapter( uint32_t index );

        virtual TCompletionCode OpenOfflineMetricsDeviceFromBuffer( uint8_t* buffer, uint32_t bufferSize, IMetricsDevice_1_15** metricsDevice );
        virtual TCompletionCode CloseOfflineMetricsDevice( IMetricsDevice_1_15* metricsDevice );
        virtual TCompletionCode SaveMetricsDeviceToBuffer( IMetricsDevice_1_15* metricsDevice, IMetricSet_1_13** metricSets, uint32_t metricSetCount, uint8_t* buffer, uint32_t* bufferSize, const uint32_t minMajorApiVersion, const uint32_t minMinorApiVersion );
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    // Class:
    //   IAdapterGroup_1_16
    //
    // Description:
    //   Abstract interface for the GPU adapters root object.
    //
    // Updates:
    // - GetAdapter:                         Update to 1.16 interface
    // - OpenOfflineMetricsDeviceFromBuffer: Update to 1.16 interface
    // - CloseOfflineMetricsDevice:          Update to 1.16 interface
    // - SaveMetricsDeviceToBuffer:          Update to 1.16 interface
    //
    // New:
    // - CreateCalculationContext:           To create calculation context
    // - DestroyCalculationContext:          To destroy calculation context
    //
    ///////////////////////////////////////////////////////////////////////////////
    class IAdapterGroup_1_16 : public IAdapterGroup_1_15
    {
    public:
        // Updates.
        using IAdapterGroup_1_15::OpenOfflineMetricsDeviceFromBuffer; // To avoid hiding by 1.16 interface function
        using IAdapterGroup_1_15::CloseOfflineMetricsDevice;          // To avoid hiding by 1.16 interface function
        using IAdapterGroup_1_15::SaveMetricsDeviceToBuffer;          // To avoid hiding by 1.16 interface function

        virtual IAdapter_1_16* GetAdapter( uint32_t index );

        virtual TCompletionCode OpenOfflineMetricsDeviceFromBuffer( uint8_t* buffer, uint32_t bufferSize, IMetricsDevice_1_16** metricsDevice );
        virtual TCompletionCode CloseOfflineMetricsDevice( IMetricsDevice_1_16* metricsDevice );
        virtual TCompletionCode SaveMetricsDeviceToBuffer( IMetricsDevice_1_16* metricsDevice, IMetricSet_1_16** metricSets, uint32_t metricSetCount, uint8_t* buffer, uint32_t* bufferSize, const uint32_t minMajorApiVersion, const uint32_t minMinorApiVersion );

        // New.
        virtual TCompletionCode CreateCalculationContext( TCalculationContextDescriptor_1_16* calculationDescriptor, ICalculationContext_1_16** calculationContext );
        virtual TCompletionCode DestroyCalculationContext( ICalculationContext_1_16* calculationContext );
    };

    //////////////////////////////////////////////////////////////////////////////////
    // Latest interfaces and typedef structs versions:
    //////////////////////////////////////////////////////////////////////////////////
    using IAdapterGroupLatest                         = IAdapterGroup_1_16;
    using IAdapterLatest                              = IAdapter_1_16;
    using ICalculationContextLatest                   = ICalculationContext_1_16;
    using IConcurrentGroupLatest                      = IConcurrentGroup_1_16;
    using IEquationLatest                             = IEquation_1_0;
    using IInformationLatest                          = IInformation_1_0;
    using IMetricEnumeratorLatest                     = IMetricEnumerator_1_13;
    using IMetricLatest                               = IMetric_1_13;
    using IMetricPrototypeLatest                      = IMetricPrototype_1_13;
    using IMetricSetLatest                            = IMetricSet_1_16;
    using IMetricsDeviceLatest                        = IMetricsDevice_1_16;
    using IOverrideLatest                             = IOverride_1_2;
    using TAdapterGroupParamsLatest                   = TAdapterGroupParams_1_6;
    using TAdapterIdLatest                            = TAdapterId_1_6;
    using TAdapterIdLuidLatest                        = TAdapterIdLuid_1_6;
    using TAdapterIdMajorMinorLatest                  = TAdapterIdMajorMinor_1_6;
    using TAdapterParamsLatest                        = TAdapterParams_1_9;
    using TApiSpecificIdLatest                        = TApiSpecificId_1_0;
    using TApiVersionLatest                           = TApiVersion_1_0;
    using TByteArrayLatest                            = TByteArray_1_0;
    using TCalculationContextDescriptorLatest         = TCalculationContextDescriptor_1_16;
    using TCalculationContextIoStreamDescriptorLatest = TCalculationContextIoStreamDescriptor_1_16;
    using TCalculationContextParamsLatest             = TCalculationContextParams_1_16;
    using TCalculationContextQueryDescriptorLatest    = TCalculationContextQueryDescriptor_1_16;
    using TConcurrentGroupParamsLatest                = TConcurrentGroupParams_1_13;
    using TDeltaFunctionLatest                        = TDeltaFunction_1_0;
    using TEngineIdClassInstanceLatest                = TEngineIdClassInstance_1_9;
    using TEngineIdLatest                             = TEngineId_1_9;
    using TEngineParamsLatest                         = TEngineParams_1_13;
    using TEquationElementLatest                      = TEquationElement_1_0;
    using TGlobalSymbolLatest                         = TGlobalSymbol_1_0;
    using TInformationParamsLatest                    = TInformationParams_1_0;
    using TMetricParamsLatest                         = TMetricParams_1_13;
    using TMetricPrototypeOptionDescriptorLatest      = TMetricPrototypeOptionDescriptor_1_13;
    using TMetricPrototypeParamsLatest                = TMetricPrototypeParams_1_13;
    using TMetricSetParamsLatest                      = TMetricSetParams_1_16;
    using TMetricsDeviceParamsLatest                  = TMetricsDeviceParams_1_2;
    using TOverrideParamsLatest                       = TOverrideParams_1_2;
    using TReadParamsLatest                           = TReadParams_1_0;
    using TSetDriverOverrideParamsLatest              = TSetDriverOverrideParams_1_2;
    using TSetFrequencyOverrideParamsLatest           = TSetFrequencyOverrideParams_1_2;
    using TSetOverrideParamsLatest                    = TSetOverrideParams_1_2;
    using TSetQueryOverrideParamsLatest               = TSetQueryOverrideParams_1_2;
    using TSubDeviceParamsLatest                      = TSubDeviceParams_1_9;
    using TTimeWindowLatest                           = TTimeWindow_1_16;
    using TTypedValueLatest                           = TTypedValue_1_0;
    using TValidValueLatest                           = TValidValue_1_13;

#ifdef __cplusplus
    extern "C"
    {
#endif

        // [Current] Factory functions
        typedef TCompletionCode( MD_STDCALL* OpenAdapterGroup_fn )( IAdapterGroupLatest** adapterGroup );

        // [Legacy] Factory functions
        typedef TCompletionCode( MD_STDCALL* OpenMetricsDevice_fn )( IMetricsDeviceLatest** metricsDevice );
        typedef TCompletionCode( MD_STDCALL* OpenMetricsDeviceFromFile_fn )( const char* fileName, void* openParams, IMetricsDeviceLatest** metricsDevice );
        typedef TCompletionCode( MD_STDCALL* CloseMetricsDevice_fn )( IMetricsDeviceLatest* metricsDevice );
        typedef TCompletionCode( MD_STDCALL* SaveMetricsDeviceToFile_fn )( const char* fileName, void* saveParams, IMetricsDeviceLatest* metricsDevice );

#ifdef __cplusplus
    }
#endif

}; // namespace MetricsDiscovery
