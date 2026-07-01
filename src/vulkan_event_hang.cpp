// Companion to vulkan_compute_wait: that test shows a host wait is *necessary* to preserve sync.
// This one shows the opposite hazard — an *unbounded* wait-for-VK_EVENT_SET on replay can hang.
//
// A device set/reset pair on one event, observed by an unsynchronized host vkGetEventStatus, is a
// data race: capture may catch the transient VK_EVENT_SET, replay may see the terminal VK_EVENT_RESET.
// A replayer that loops until the captured VK_EVENT_SET reappears then spins forever (event is durably
// reset, nothing re-signals it). This is a race, so it cannot be deterministic — if it were, capture
// and replay would agree and there would be no hang. We just arm the capture (record SET polls) and
// let a looped replay hit the hang.

#include "vulkan_common.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t kDefaultIterations   = 20000;
constexpr uint32_t kDefaultFillerCopies = 6;          // delays the device reset so capture can catch SET
constexpr VkDeviceSize kFillerBytes     = 1u << 20;   // 1 MiB per filler copy

struct Buffer
{
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
};

static Buffer create_device_buffer(const vulkan_setup_t& vulkan, VkDeviceSize size, const char* name)
{
	Buffer out;
	VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr };
	bi.size        = size;
	bi.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	check(vkCreateBuffer(vulkan.device, &bi, nullptr, &out.buffer));
	test_set_name(vulkan, VK_OBJECT_TYPE_BUFFER, (uint64_t)out.buffer, name);

	VkMemoryRequirements req = {};
	vkGetBufferMemoryRequirements(vulkan.device, out.buffer, &req);
	VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr };
	ai.allocationSize  = req.size;
	ai.memoryTypeIndex = get_device_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	check(vkAllocateMemory(vulkan.device, &ai, nullptr, &out.memory));
	check(vkBindBufferMemory(vulkan.device, out.buffer, out.memory, 0));
	return out;
}

static void destroy_buffer(const vulkan_setup_t& vulkan, Buffer& b)
{
	if (b.buffer) vkDestroyBuffer(vulkan.device, b.buffer, nullptr);
	if (b.memory) testFreeMemory(vulkan, b.memory);
	b = {};
}

static void show_usage()
{
	printf("-it/--iterations N     set/reset/poll iterations to record (default %u)\n", kDefaultIterations);
	printf("-fc/--filler-copies N  transfer copies delaying the device reset (default %u)\n", kDefaultFillerCopies);
}

static bool test_cmdopt(int& i, int argc, char** argv, vulkan_req_t& reqs)
{
	if (match(argv[i], "-it", "--iterations"))    { reqs.options["iterations"]    = get_arg(argv, ++i, argc); return true; }
	if (match(argv[i], "-fc", "--filler-copies")) { reqs.options["filler_copies"] = get_arg(argv, ++i, argc); return true; }
	return false;
}

int main(int argc, char** argv)
{
	vulkan_req_t reqs;
	reqs.required_queue_flags = VK_QUEUE_COMPUTE_BIT; // needs events + transfer
	reqs.options["iterations"]    = static_cast<int>(kDefaultIterations);
	reqs.options["filler_copies"] = static_cast<int>(kDefaultFillerCopies);
	reqs.usage  = show_usage;
	reqs.cmdopt = test_cmdopt;

	vulkan_setup_t vulkan = test_init(argc, argv, "vulkan_event_hang", reqs);

	const uint32_t iterations    = static_cast<uint32_t>(std::get<int>(reqs.options.at("iterations")));
	const uint32_t filler_copies = static_cast<uint32_t>(std::get<int>(reqs.options.at("filler_copies")));

	VkQueue queue = VK_NULL_HANDLE;
	vkGetDeviceQueue(vulkan.device, vulkan.queue_family_index, 0, &queue);

	VkEventCreateInfo ei = { VK_STRUCTURE_TYPE_EVENT_CREATE_INFO, nullptr };
	VkEvent event = VK_NULL_HANDLE;
	check(vkCreateEvent(vulkan.device, &ei, nullptr, &event));
	test_set_name(vulkan, VK_OBJECT_TYPE_EVENT, (uint64_t)event, "event_hang_event");

	Buffer src = create_device_buffer(vulkan, kFillerBytes, "event_hang_src");
	Buffer dst = create_device_buffer(vulkan, kFillerBytes, "event_hang_dst");

	VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr };
	pci.queueFamilyIndex = vulkan.queue_family_index;
	VkCommandPool pool = VK_NULL_HANDLE;
	check(vkCreateCommandPool(vulkan.device, &pci, nullptr, &pool));

	std::array<VkCommandBuffer, 2> cmd = {};
	VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr };
	cbi.commandPool        = pool;
	cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = cmd.size();
	check(vkAllocateCommandBuffers(vulkan.device, &cbi, cmd.data()));
	VkCommandBuffer cmd_set   = cmd[0];
	VkCommandBuffer cmd_reset = cmd[1];

	// cmd_set: signal the event as early as possible (completes quickly).
	VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr };
	check(vkBeginCommandBuffer(cmd_set, &begin));
	vkCmdSetEvent(cmd_set, event, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
	check(vkEndCommandBuffer(cmd_set));

	// cmd_reset: burn some transfer time (serialized copies), then reset the event. The filler widens
	// the window in which the event is SET, so the capture-side poll can catch it.
	check(vkBeginCommandBuffer(cmd_reset, &begin));
	VkBufferCopy region = { 0, 0, kFillerBytes };
	for (uint32_t c = 0; c < filler_copies; c++)
	{
		vkCmdCopyBuffer(cmd_reset, src.buffer, dst.buffer, 1, &region);
		VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr };
		mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd_reset, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
		                     nullptr, 0, nullptr);
	}
	vkCmdResetEvent(cmd_reset, event, VK_PIPELINE_STAGE_TRANSFER_BIT);
	check(vkEndCommandBuffer(cmd_reset));

	uint64_t observed_set = 0, observed_reset = 0;
	for (uint32_t it = 0; it < iterations; it++)
	{
		VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr };
		submit.commandBufferCount = 1;

		submit.pCommandBuffers = &cmd_set;
		check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
		submit.pCommandBuffers = &cmd_reset;
		check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));

		// The captured call. Unsynchronized: races the device set -> reset transition.
		VkResult status = vkGetEventStatus(vulkan.device, event);
		if (status == VK_EVENT_SET) observed_set++;
		else                        observed_reset++;

		check(vkQueueWaitIdle(queue)); // settle before reusing the event/command buffers
	}

	printf("vulkan_event_hang: iterations=%u filler_copies=%u -> observed VK_EVENT_SET=%lu, VK_EVENT_RESET=%lu\n",
	       iterations, filler_copies, (unsigned long)observed_set, (unsigned long)observed_reset);
	if (observed_set == 0)
	{
		printf("  WARNING: no SET observed -> capture is not armed. Increase --filler-copies and/or --iterations.\n");
	}
	else
	{
		printf("  capture armed: %lu recorded vkGetEventStatus==VK_EVENT_SET call(s). Replay in a loop; an\n"
		       "  unbounded wait-for-SET replayer will hang on a run where replay observes the terminal RESET.\n",
		       (unsigned long)observed_set);
	}

	vkDestroyCommandPool(vulkan.device, pool, nullptr);
	destroy_buffer(vulkan, dst);
	destroy_buffer(vulkan, src);
	vkDestroyEvent(vulkan.device, event, nullptr);
	test_done(vulkan);
	return 0;
}
