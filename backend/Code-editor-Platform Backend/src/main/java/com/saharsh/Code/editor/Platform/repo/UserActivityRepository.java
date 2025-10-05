package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.UserActivity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserActivityRepository extends JpaRepository<UserActivity, Long> {
    List<UserActivity> findByUser_IdAndTest_TestId(Integer userId, Integer testId);

}