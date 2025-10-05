package com.saharsh.Code.editor.Platform.controller;

import com.saharsh.Code.editor.Platform.Dto.StudentTestsDTO;
import com.saharsh.Code.editor.Platform.model.TestAttempt;
import com.saharsh.Code.editor.Platform.model.Users;
import com.saharsh.Code.editor.Platform.repo.TestAttemptRepository;
import com.saharsh.Code.editor.Platform.repo.UsersRepository;
import com.saharsh.Code.editor.Platform.service.TestAttemptService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/testattempts")
public class TestAttemptController {

    @Autowired
    private TestAttemptService testAttemptService;

    @Autowired
    private TestAttemptRepository repo;
    // Get all test attempts
    @GetMapping
    public ResponseEntity<List<TestAttempt>> getAllTestAttempts() {
        List<TestAttempt> testAttempts = testAttemptService.getAllTestAttempts();
        return new ResponseEntity<>(testAttempts, HttpStatus.OK);
    }

    //    // Get test attempt by ID
//    @GetMapping("/{id}")
//    public List<TestAttempt> getTestAttemptById(@PathVariable int id) {
//        List<TestAttempt> testAttempt = repo.findByTest_TestId(id);
//        return testAttempt;
//    }
    @Autowired
    private UsersRepository  Userrepo;

    @GetMapping("/process")
    public String processTests() {
        return "Processing tests...";
    }

    @GetMapping("/{id}")
    public List<StudentTestsDTO> getTestAttemptById(@PathVariable Long id) {
        System.out.println(id);

        List<TestAttempt> testAttempt = repo.findByTest_TestId(id);
//        System.out.println(testAttempt);
        List<StudentTestsDTO> ans = new ArrayList<>();
        for(TestAttempt x : testAttempt){
            StudentTestsDTO obj = new StudentTestsDTO();
            Users temp = Userrepo.findByUsername(x.getUsername());
            obj.setUserId(temp.getId());
            obj.setScore(x.getTotalScore());
            obj.setId(x.getId());
            obj.setTestId(x.getTest().getTestId());
            ans.add(obj);
        }

        return ans;
    }

    //     Create a new test attempt
    @PostMapping
    public ResponseEntity<TestAttempt> createTestAttempt(@RequestBody TestAttempt testAttempt) {
        TestAttempt savedTestAttempt = testAttemptService.saveTestAttempt(testAttempt);
        return new ResponseEntity<>(savedTestAttempt, HttpStatus.CREATED);
    }

//    @PostMapping
//    public String createTestAttempt(@RequestBody TestAttempt testAttempt) {
//        System.out.println(testAttempt);
//       return "SUcess";
//    }

    // Update an existing test attempt
    @PutMapping("/{id}")
    public ResponseEntity<TestAttempt> updateTestAttempt(@PathVariable int id, @RequestBody TestAttempt testAttempt) {
        Optional<TestAttempt> existingTestAttempt = testAttemptService.getTestAttemptById(id);
        if (existingTestAttempt.isPresent()) {
            testAttempt.setId(id); // Ensure the ID from the path is used
            TestAttempt updatedTestAttempt = testAttemptService.saveTestAttempt(testAttempt);
            return new ResponseEntity<>(updatedTestAttempt, HttpStatus.OK);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    // Delete a test attempt by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTestAttempt(@PathVariable int id) {
        if (testAttemptService.getTestAttemptById(id).isPresent()) {
            testAttemptService.deleteTestAttempt(id);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }
}