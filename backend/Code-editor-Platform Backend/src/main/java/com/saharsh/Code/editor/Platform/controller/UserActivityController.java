package com.saharsh.Code.editor.Platform.controller;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.saharsh.Code.editor.Platform.Dto.LogDTO;
import com.saharsh.Code.editor.Platform.Dto.UserActivityDTO;
import com.saharsh.Code.editor.Platform.model.Test;
import com.saharsh.Code.editor.Platform.model.UserActivity;
import com.saharsh.Code.editor.Platform.model.Users;
import com.saharsh.Code.editor.Platform.repo.TestRepository;
import com.saharsh.Code.editor.Platform.repo.UserActivityRepository;
import com.saharsh.Code.editor.Platform.repo.UsersRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/activities")
@RequiredArgsConstructor
public class UserActivityController {
    private final UserActivityRepository userActivityRepository;
    private final UsersRepository usersRepository;
    private final TestRepository testRepository;
    private final ObjectMapper objectMapper; // inject ObjectMapper

    @PostMapping("/save")
    public List<UserActivity> saveActivities(@RequestBody UserActivityDTO dto) throws Exception {
        Users user = usersRepository.findById(dto.getUserId())
                .orElseThrow(() -> new RuntimeException("User not found"));

        Test test = testRepository.findById(Integer.parseInt(dto.getTestId()))
                .orElseThrow(() -> new RuntimeException("Test not found"));

        List<UserActivity> activities = new ArrayList<>();

        for (LogDTO log : dto.getLogs()) {
            UserActivity ua = new UserActivity();
            ua.setUser(user);
            ua.setTest(test);
            ua.setType(log.getType());
            ua.setPhase(log.getPhase());
            if (log.getTs() != null) {
                ua.setTs(Instant.parse(log.getTs()));
            }
            ua.setDuration(log.getDuration());
            ua.setDetails(objectMapper.writeValueAsString(log.getDetails())); // serialize JSON object
            activities.add(ua);
        }

        return userActivityRepository.saveAll(activities);
    }

    @GetMapping("/all")
    public List<UserActivity> getAllActivities() {
        return userActivityRepository.findAll();
    }

    @GetMapping("/user/{userId}/test/{testId}")
    @Transactional(readOnly = true)
    public List<LogDTO> getActivitiesByUserAndTest(@PathVariable Integer userId,
                                                   @PathVariable Integer testId) {
        List<UserActivity> as =  userActivityRepository.findByUser_IdAndTest_TestId(userId, testId);
        List<LogDTO> ans = new ArrayList<>();
        for(UserActivity x : as){
            LogDTO obj = new LogDTO();
            obj.setType(x.getType());
            obj.setDuration(x.getDuration());
            obj.setTs(x.getTs().toString());
            obj.setPhase(x.getPhase());
            ans.add(obj);
        }
        return ans;
    }
}
