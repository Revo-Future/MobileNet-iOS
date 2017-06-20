import Foundation

/*
 Encapsulates access to the weights that are stored in parameters.data.
 
 We only need to read from the parameters file while the neural network is
 being created. The weights are copied into the network (as 16-bit floats),
 so once the network is set up we no longer need to keep parameters.data
 in memory.
 
 Because this is a huge file, we use mmap() so that not the entire file has
 to be read into memory at once. Deallocating VGGNetData unmaps the file.
 */
class MobileNetsData {
    // Size of the data file in bytes.
    let fileSize = 16884128
    
    // These are the offsets in the big blob of data of the weights and biases
    // for each layer.
    
    var conv1_s2_w: UnsafeMutablePointer<Float> { return ptr + 0 }
    var conv1_s2_b: UnsafeMutablePointer<Float> { return ptr + 864 }
    var conv2_1_dw_w: UnsafeMutablePointer<Float> { return ptr + 896 }
    var conv2_1_dw_b: UnsafeMutablePointer<Float> { return ptr + 1184 }
    var conv2_1_s1_w: UnsafeMutablePointer<Float> { return ptr + 1216 }
    var conv2_1_s1_b: UnsafeMutablePointer<Float> { return ptr + 3264 }
    var conv2_2_dw_w: UnsafeMutablePointer<Float> { return ptr + 3328 }
    var conv2_2_dw_b: UnsafeMutablePointer<Float> { return ptr + 3904 }
    var conv2_2_s1_w: UnsafeMutablePointer<Float> { return ptr + 3968 }
    var conv2_2_s1_b: UnsafeMutablePointer<Float> { return ptr + 12160 }
    var conv3_1_dw_w: UnsafeMutablePointer<Float> { return ptr + 12288 }
    var conv3_1_dw_b: UnsafeMutablePointer<Float> { return ptr + 13440 }
    var conv3_1_s1_w: UnsafeMutablePointer<Float> { return ptr + 13568 }
    var conv3_1_s1_b: UnsafeMutablePointer<Float> { return ptr + 29952 }
    var conv3_2_dw_w: UnsafeMutablePointer<Float> { return ptr + 30080 }
    var conv3_2_dw_b: UnsafeMutablePointer<Float> { return ptr + 31232 }
    var conv3_2_s1_w: UnsafeMutablePointer<Float> { return ptr + 31360 }
    var conv3_2_s1_b: UnsafeMutablePointer<Float> { return ptr + 64128 }
    var conv4_1_dw_w: UnsafeMutablePointer<Float> { return ptr + 64384 }
    var conv4_1_dw_b: UnsafeMutablePointer<Float> { return ptr + 66688 }
    var conv4_1_s1_w: UnsafeMutablePointer<Float> { return ptr + 66944 }
    var conv4_1_s1_b: UnsafeMutablePointer<Float> { return ptr + 132480 }
    var conv4_2_dw_w: UnsafeMutablePointer<Float> { return ptr + 132736 }
    var conv4_2_dw_b: UnsafeMutablePointer<Float> { return ptr + 135040 }
    var conv4_2_s1_w: UnsafeMutablePointer<Float> { return ptr + 135296 }
    var conv4_2_s1_b: UnsafeMutablePointer<Float> { return ptr + 266368 }
    var conv5_1_dw_w: UnsafeMutablePointer<Float> { return ptr + 266880 }
    var conv5_1_dw_b: UnsafeMutablePointer<Float> { return ptr + 271488 }
    var conv5_1_s1_w: UnsafeMutablePointer<Float> { return ptr + 272000 }
    var conv5_1_s1_b: UnsafeMutablePointer<Float> { return ptr + 534144 }
    var conv5_2_dw_w: UnsafeMutablePointer<Float> { return ptr + 534656 }
    var conv5_2_dw_b: UnsafeMutablePointer<Float> { return ptr + 539264 }
    var conv5_2_s1_w: UnsafeMutablePointer<Float> { return ptr + 539776 }
    var conv5_2_s1_b: UnsafeMutablePointer<Float> { return ptr + 801920 }
    var conv5_3_dw_w: UnsafeMutablePointer<Float> { return ptr + 802432 }
    var conv5_3_dw_b: UnsafeMutablePointer<Float> { return ptr + 807040 }
    var conv5_3_s1_w: UnsafeMutablePointer<Float> { return ptr + 807552 }
    var conv5_3_s1_b: UnsafeMutablePointer<Float> { return ptr + 1069696 }
    var conv5_4_dw_w: UnsafeMutablePointer<Float> { return ptr + 1070208 }
    var conv5_4_dw_b: UnsafeMutablePointer<Float> { return ptr + 1074816 }
    var conv5_4_s1_w: UnsafeMutablePointer<Float> { return ptr + 1075328 }
    var conv5_4_s1_b: UnsafeMutablePointer<Float> { return ptr + 1337472 }
    var conv5_5_dw_w: UnsafeMutablePointer<Float> { return ptr + 1337984 }
    var conv5_5_dw_b: UnsafeMutablePointer<Float> { return ptr + 1342592 }
    var conv5_5_s1_w: UnsafeMutablePointer<Float> { return ptr + 1343104 }
    var conv5_5_s1_b: UnsafeMutablePointer<Float> { return ptr + 1605248 }
    var conv5_6_dw_w: UnsafeMutablePointer<Float> { return ptr + 1605760 }
    var conv5_6_dw_b: UnsafeMutablePointer<Float> { return ptr + 1610368 }
    var conv5_6_s1_w: UnsafeMutablePointer<Float> { return ptr + 1610880 }
    var conv5_6_s1_b: UnsafeMutablePointer<Float> { return ptr + 2135168 }
    var conv6_1_dw_w: UnsafeMutablePointer<Float> { return ptr + 2136192 }
    var conv6_1_dw_b: UnsafeMutablePointer<Float> { return ptr + 2145408 }
    var conv6_1_s1_w: UnsafeMutablePointer<Float> { return ptr + 2146432 }
    var conv6_1_s1_b: UnsafeMutablePointer<Float> { return ptr + 3195008 }
    var fc7_w: UnsafeMutablePointer<Float> { return ptr + 3196032 }
    var fc7_b: UnsafeMutablePointer<Float> { return ptr + 4220032 }
    
    
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private var ptr: UnsafeMutablePointer<Float>!
    
    /* This is for debugging. Initializing the weights to 0 gives an output of
     0.000999451, or approx 1/1000 for all classes, which is what you'd expect
     for a softmax classifier. */
    init() {
        let numBytes = fileSize / MemoryLayout<Float>.size
        ptr = UnsafeMutablePointer<Float>.allocate(capacity: numBytes)
        ptr.initialize(to: 0, count: numBytes)
    }
    
    init?(path: String) {
        
        fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        if fd == -1 {
            print("Error: failed to open \"\(path)\", error = \(errno)")
            return nil
        }
        
        hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        if hdr == nil {
            print("Error: mmap failed, errno = \(errno)")
            return nil
        }
        
        let numBytes = fileSize / MemoryLayout<Float>.size
        ptr = hdr.bindMemory(to: Float.self, capacity: numBytes)
        if ptr == UnsafeMutablePointer<Float>(bitPattern: -1) {
            print("Error: mmap failed, errno = \(errno)")
            return nil
        }
        
    }
    
    deinit{
        print("deinit \(self)")
        
        if let hdr = hdr {
            let result = munmap(hdr, Int(fileSize))
            assert(result == 0, "Error: munmap failed, errno = \(errno)")
        }
        if let fd = fd {
            close(fd)
        }
    }
}
