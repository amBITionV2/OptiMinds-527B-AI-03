package com.saharsh.Code.editor.Platform.controller;
import com.saharsh.Code.editor.Platform.Dto.CreateTestRequest;
import com.saharsh.Code.editor.Platform.Dto.GetTestsDto;
import com.saharsh.Code.editor.Platform.model.Test;
import com.saharsh.Code.editor.Platform.service.TestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/tests")
public class TestController {

    @Autowired
    private TestService testService;

    // Get all tests
    @GetMapping
    public ResponseEntity<List<GetTestsDto>> getAllTests() {
        List<Test> tests = testService.getAllTests();

        List<GetTestsDto> ans = new ArrayList<>();
        for(Test test : tests){
            GetTestsDto obj = new GetTestsDto();
            obj.setTestId(test.getTestId());
            obj.setTestName(test.getTestName());
            obj.setCreatedAt(test.getCreatedAt());
            obj.setEndTime(test.getEndTime());
            obj.setStartTime(test.getStartTime());
            ans.add(obj);
        }
        return new ResponseEntity<>(ans, HttpStatus.OK);
    }


    // Get test by ID
    @GetMapping("/{id}")
    public ResponseEntity<Test> getTestById(@PathVariable int id) {
        Optional<Test> test = testService.getTestById(id);
        return test.map(value -> new ResponseEntity<>(value, HttpStatus.OK))
                .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }

    // Create a new test
    @PostMapping
    public ResponseEntity<Test> createTest(@RequestBody CreateTestRequest request) {
        Test savedTest = testService.saveTest(request);
        return new ResponseEntity<>(savedTest, HttpStatus.CREATED);
    }

//    // Update an existing test
//    @PutMapping("/{id}")
//    public ResponseEntity<Test> updateTest(@PathVariable int id, @RequestBody Test test) {
//        Optional<Test> existingTest = testService.getTestById(id);
//        if (existingTest.isPresent()) {
//            test.setTestId(id); // Ensure the ID from the path is used
//            Test updatedTest = testService.saveTest(test);
//            return new ResponseEntity<>(updatedTest, HttpStatus.OK);
//        } else {
//            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
//        }
//    }

    // Delete a test by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTest(@PathVariable int id) {
        if (testService.getTestById(id).isPresent()) {
            testService.deleteTest(id);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }
    @PostMapping("/upload")
    public ResponseEntity<String> uploadImage(
            @RequestParam("userId") Long userId,
            @RequestParam("testId") Long testId,
            @RequestParam("frame") MultipartFile frame,
            @RequestParam("time") Long time,
            @RequestParam("device") boolean phone
    ) {
        try {
            testService.saveImage(userId, testId, frame,time,phone);
            return ResponseEntity.ok("Image saved successfully");
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Failed to save image: " + e.getMessage());
        }
    }

}