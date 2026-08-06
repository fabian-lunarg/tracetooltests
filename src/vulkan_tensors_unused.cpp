// Trim/optimize testcase: three tensors are created and bound, but only two of them are
// ever referenced by a command buffer. The third exists purely so that gfxrecon-optimize
// has an unreferenced tensor to strip the InitTensorCommand block for.

#include "vulkan_common.h"
#include <inttypes.h>
#include <cstring>

int main(int argc, char** argv)
{
	vulkan_req_t reqs;
	VkPhysicalDeviceTensorFeaturesARM tensor_features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM, nullptr };
	tensor_features.tensors = VK_TRUE;
	reqs.device_extensions.push_back("VK_ARM_tensors");
	reqs.extension_features = (VkBaseInStructure*)&tensor_features;
	reqs.minApiVersion = VK_API_VERSION_1_3;
	reqs.apiVersion = VK_API_VERSION_1_3;
	vulkan_setup_t vulkan = test_init(argc, argv, "vulkan_tensors_unused", reqs);
	VkResult r;

	bench_start_iteration(vulkan.bench);

	MAKEDEVICEPROCADDR(vulkan, vkCreateTensorARM);
	MAKEDEVICEPROCADDR(vulkan, vkDestroyTensorARM);
	MAKEDEVICEPROCADDR(vulkan, vkGetTensorMemoryRequirementsARM);
	MAKEDEVICEPROCADDR(vulkan, vkCreateTensorViewARM);
	MAKEDEVICEPROCADDR(vulkan, vkDestroyTensorViewARM);
	MAKEDEVICEPROCADDR(vulkan, vkBindTensorMemoryARM);
	MAKEDEVICEPROCADDR(vulkan, vkCmdCopyTensorARM);

	VkQueue queue;
	vkGetDeviceQueue(vulkan.device, 0, 0, &queue);

	std::vector<int64_t> dimensions { 64, 64 };

	VkTensorDescriptionARM td = { VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM, nullptr };
	td.tiling = VK_TENSOR_TILING_LINEAR_ARM;
	td.format = VK_FORMAT_R8_UINT;
	td.dimensionCount = dimensions.size();
	td.pDimensions = dimensions.data();
	td.pStrides = nullptr;
	td.usage = VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM | VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM | VK_TENSOR_USAGE_SHADER_BIT_ARM;

	VkTensorCreateInfoARM tci = { VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM, nullptr };
	tci.flags = 0;
	tci.pDescription = &td;
	tci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	tci.queueFamilyIndexCount = 0;
	tci.pQueueFamilyIndices = nullptr;

	VkTensorARM tensor = VK_NULL_HANDLE;  // copy source, referenced
	VkTensorARM target = VK_NULL_HANDLE;  // copy destination, referenced
	VkTensorARM unused = VK_NULL_HANDLE;  // never referenced by any command
	r = pf_vkCreateTensorARM(vulkan.device, &tci, nullptr, &tensor);
	check(r);
	r = pf_vkCreateTensorARM(vulkan.device, &tci, nullptr, &target);
	check(r);
	r = pf_vkCreateTensorARM(vulkan.device, &tci, nullptr, &unused);
	check(r);

	VkMemoryRequirements2 memreq = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr };
	VkTensorMemoryRequirementsInfoARM tmri = { VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM, nullptr };
	tmri.tensor = tensor;
	pf_vkGetTensorMemoryRequirementsARM(vulkan.device, &tmri, &memreq);

	const uint32_t memoryTypeIndex = get_device_memory_type(memreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	const uint32_t align_mod = memreq.memoryRequirements.size % memreq.memoryRequirements.alignment;
	const uint32_t aligned_size = (align_mod == 0) ? memreq.memoryRequirements.size : (memreq.memoryRequirements.size + memreq.memoryRequirements.alignment - align_mod);

	VkMemoryAllocateInfo pAllocateMemInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr };
	pAllocateMemInfo.memoryTypeIndex = memoryTypeIndex;
	pAllocateMemInfo.allocationSize = aligned_size * 3;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	r = vkAllocateMemory(vulkan.device, &pAllocateMemInfo, nullptr, &memory);
	check(r);

	VkBindTensorMemoryInfoARM btmi = { VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM, nullptr };
	btmi.tensor = tensor;
	btmi.memory = memory;
	btmi.memoryOffset = 0;
	r = pf_vkBindTensorMemoryARM(vulkan.device, 1, &btmi);
	check(r);
	btmi.tensor = target;
	btmi.memoryOffset = aligned_size;
	r = pf_vkBindTensorMemoryARM(vulkan.device, 1, &btmi);
	check(r);
	btmi.tensor = unused;
	btmi.memoryOffset = aligned_size * 2;
	r = pf_vkBindTensorMemoryARM(vulkan.device, 1, &btmi);
	check(r);

	// Distinct fill per tensor, so a wrongly-stripped or mixed-up InitTensorCommand is visible
	// in the capture rather than silently plausible.
	char* mapped = nullptr;
	r = vkMapMemory(vulkan.device, memory, 0, VK_WHOLE_SIZE, 0, (void**)&mapped);
	check(r);
	memset(mapped, 0x11, aligned_size);
	memset(mapped + aligned_size, 0x22, aligned_size);
	memset(mapped + aligned_size * 2, 0x33, aligned_size);
	vkUnmapMemory(vulkan.device, memory);

	// A view on the referenced tensor only; the unused tensor gets none.
	VkTensorViewCreateInfoARM tvci = { VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM, nullptr };
	tvci.flags = 0;
	tvci.tensor = tensor;
	tvci.format = VK_FORMAT_R8_UINT;
	VkTensorViewARM tensor_view = VK_NULL_HANDLE;
	r = pf_vkCreateTensorViewARM(vulkan.device, &tvci, nullptr, &tensor_view);
	check(r);

	VkCommandPool command_pool;
	VkCommandPoolCreateInfo command_pool_create_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr };
	command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	r = vkCreateCommandPool(vulkan.device, &command_pool_create_info, NULL, &command_pool);
	check(r);

	VkCommandBuffer command_buffer;
	VkCommandBufferAllocateInfo command_buffer_allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr };
	command_buffer_allocate_info.commandPool = command_pool;
	command_buffer_allocate_info.commandBufferCount = 1;
	r = vkAllocateCommandBuffers(vulkan.device, &command_buffer_allocate_info, &command_buffer);
	check(r);

	VkFence fence;
	VkFenceCreateInfo fence_create_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	r = vkCreateFence(vulkan.device, &fence_create_info, NULL, &fence);
	check(r);

	VkCommandBufferBeginInfo command_buffer_begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr };
	command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	r = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
	check(r);

	VkTensorCopyARM tc = { VK_STRUCTURE_TYPE_TENSOR_COPY_ARM, nullptr };
	tc.dimensionCount = dimensions.size();
	tc.pSrcOffset = nullptr;
	tc.pDstOffset = nullptr;
	tc.pExtent = nullptr;
	VkCopyTensorInfoARM cti = { VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_ARM, nullptr };
	cti.srcTensor = tensor;
	cti.dstTensor = target;
	cti.regionCount = 1;
	cti.pRegions = &tc;
	pf_vkCmdCopyTensorARM(command_buffer, &cti); // 'unused' deliberately absent

	r = vkEndCommandBuffer(command_buffer);
	check(r);

	VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr };
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
	r = vkQueueSubmit(queue, 1, &submit_info, fence);
	check(r);

	r = vkWaitForFences(vulkan.device, 1, &fence, VK_TRUE, UINT64_MAX);
	check(r);

	vkDestroyFence(vulkan.device, fence, nullptr);
	vkFreeCommandBuffers(vulkan.device, command_pool, 1, &command_buffer);
	vkDestroyCommandPool(vulkan.device, command_pool, nullptr);

	pf_vkDestroyTensorViewARM(vulkan.device, tensor_view, nullptr);
	pf_vkDestroyTensorARM(vulkan.device, tensor, nullptr);
	pf_vkDestroyTensorARM(vulkan.device, target, nullptr);
	pf_vkDestroyTensorARM(vulkan.device, unused, nullptr);
	testFreeMemory(vulkan, memory);

	bench_stop_iteration(vulkan.bench);

	test_done(vulkan);

	return 0;
}
