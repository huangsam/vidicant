const std = @import("std");

fn configureOpenCV(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, custom_path: ?[]const u8) void {
    if (custom_path) |p| {
        const inc_path = b.fmt("{s}/include", .{p});
        const lib_path = b.fmt("{s}/lib", .{p});
        mod.addIncludePath(.{ .cwd_relative = inc_path });
        mod.addLibraryPath(.{ .cwd_relative = lib_path });
        return;
    }

    const os = target.result.os.tag;
    if (os == .macos) {
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/include/opencv5" });
        mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/include/opencv4" });
        mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/opencv/lib" });
    } else if (os == .linux) {
        mod.addIncludePath(.{ .cwd_relative = "/usr/include" });
        mod.addIncludePath(.{ .cwd_relative = "/usr/include/opencv4" });
        mod.addIncludePath(.{ .cwd_relative = "/usr/include/opencv5" });
        mod.addIncludePath(.{ .cwd_relative = "/usr/local/include" });
        mod.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
        mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/aarch64-linux-gnu" });
        mod.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
    } else if (os == .windows) {
        mod.addIncludePath(.{ .cwd_relative = "C:/opencv/build/include" });
        mod.addLibraryPath(.{ .cwd_relative = "C:/opencv/build/x64/vc16/lib" });
    }
}

fn configureCppStdLib(mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    mod.link_libc = true;
    if (target.result.os.tag == .linux) {
        mod.linkSystemLibrary("stdc++", .{});
    } else {
        mod.link_libcpp = true;
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const opencv_path = b.option([]const u8, "opencv-path", "Custom path to OpenCV installation directory");

    const cxx_flags: []const []const u8 = if (target.result.os.tag == .linux)
        &.{ "-std=c++17", "-Wall", "-Wextra", "-stdlib=libstdc++" }
    else
        &.{ "-std=c++17", "-Wall", "-Wextra" };

    // 1. Module and Dynamic Library for Python / C-ABI (libvidicant.dylib / .so / .dll)
    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addCSourceFiles(.{
        .files = &.{
            "src/image.cpp",
            "src/video.cpp",
            "src/controller.cpp",
            "src/vidicant_c_api.cpp",
        },
        .flags = cxx_flags,
    });
    lib_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, lib_mod, target, opencv_path);

    lib_mod.linkSystemLibrary("opencv_core", .{});
    lib_mod.linkSystemLibrary("opencv_imgproc", .{});
    lib_mod.linkSystemLibrary("opencv_imgcodecs", .{});
    lib_mod.linkSystemLibrary("opencv_video", .{});
    lib_mod.linkSystemLibrary("opencv_videoio", .{});
    lib_mod.linkSystemLibrary("opencv_objdetect", .{});
    lib_mod.linkSystemLibrary("opencv_dnn", .{});
    configureCppStdLib(lib_mod, target);

    const lib = b.addLibrary(.{
        .name = "vidicant",
        .linkage = .dynamic,
        .root_module = lib_mod,
    });

    b.installArtifact(lib);

    // 2. Module and Executable for CLI (vidicant_cli)
    const exe_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addCSourceFiles(.{
        .files = &.{
            "src/main.cpp",
            "src/controller.cpp",
            "src/image.cpp",
            "src/video.cpp",
        },
        .flags = cxx_flags,
    });
    exe_mod.addIncludePath(.{ .cwd_relative = "include" });
    configureOpenCV(b, exe_mod, target, opencv_path);

    exe_mod.linkSystemLibrary("opencv_core", .{});
    exe_mod.linkSystemLibrary("opencv_imgproc", .{});
    exe_mod.linkSystemLibrary("opencv_imgcodecs", .{});
    exe_mod.linkSystemLibrary("opencv_video", .{});
    exe_mod.linkSystemLibrary("opencv_videoio", .{});
    exe_mod.linkSystemLibrary("opencv_objdetect", .{});
    exe_mod.linkSystemLibrary("opencv_dnn", .{});
    configureCppStdLib(exe_mod, target);

    const exe = b.addExecutable(.{
        .name = "vidicant_cli",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);
}
