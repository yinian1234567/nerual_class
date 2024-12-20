import time
from psychopy import visual, event, core, sound
from moviepy.editor import VideoFileClip

reaction = []
answer = []
# correct = ['1','2','3','1','3','3','2','4','2','2','1','1','1','1']
correct = ['1','2','3','1','3','3']

def reaction_time_test(audio_path, vision_path, win):
    event.clearEvents()
    first_key, first_key_time = '', 0.

    audio = sound.Sound(audio_path)
    duration = min(VideoFileClip(vision_path).duration, audio.getDuration())
    start_time = time.time()
    video = visual.MovieStim(win, vision_path, loop=False)
    video.seek(4)
    audio.play()
    while time.time() - start_time <= duration:
        video.draw()
        win.flip()
        # 检测按键
        keys = event.getKeys()
        if keys:
            first_key = keys[0]
            if first_key in ['1', '2', '3', '4', '5']:
                react_duration = time.time() - start_time
                reaction.append(react_duration)
                answer.append(first_key)
                break
    if first_key == '' and first_key_time == 0.0:
        reaction.append(0)
        answer.append('0')
    return


def main():
    # timer = core.Clock()

    win = visual.Window([1200, 800], waitBlanking=True, color=[-1, -1, -1])

    text = visual.TextStim(
        win, text="您好，欢迎来到我们的实验！\n按下任意键继续...", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    event.waitKeys()

    text = "实验正式开始\n在实验中您将看到一段视频并听到一段录音\n请您在观看过程中立刻做出决策\n实验分为两个部分\n第一部分在每次实验中都会为您提供不同的选项\n请您根据相应选项要求进行选择\n如果您已经清楚该部分要求\n按下任意键继续实验..."
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()
    event.waitKeys()

    '''
    第一部分实验过程
    '''
    text = "在本次实验中，\n请从猫、非猫两个选项之间进行选择，\n猫请按1，非猫请按2，\n如果您已经清楚按键，\n请依次按下非猫、猫以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(0, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\cat\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\cat\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    text = "在本次实验中，\n请从狗、狐狸、以上均不是三个选项之间进行选择，\n狗请按1，狐狸请按2，两个都不是请按3，\n如果您已经清楚按键，\n请依次按下狐狸、狗、都不是以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1', '3']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\fox\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\fox\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    text = "在本次实验中，\n请从狗、狐狸、以上均不是三个选项之间进行选择，\n狗请按1，狐狸请按2，两个都不是请按3，\n如果您已经清楚按键，\n请依次按下狐狸、狗、都不是以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1', '3']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\cat\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\cat\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    text = "在本次实验中，\n请从狗、狐狸、以上均不是三个选项之间进行选择，\n狗请按1，狐狸请按2，两个都不是请按3，\n如果您已经清楚按键，\n请依次按下狐狸、狗、都不是以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1', '3']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\dog\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\dog\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    text = "在本次实验中，\n请从狗、狐狸、以上均不是三个选项之间进行选择，\n狗请按1，狐狸请按2，两个都不是请按3，\n如果您已经清楚按键，\n请依次按下狐狸、狗、都不是以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1', '3']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\deer\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\deer\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    text = "在本次实验中，\n请从狗、狐狸、以上均不是三个选项之间进行选择，\n狗请按1，狐狸请按2，两个都不是请按3，\n如果您已经清楚按键，\n请依次按下狐狸、狗、都不是以开始实验"
    text = visual.TextStim(win,
                           text=text,
                           color=[1, 1, 1],
                           pos=(-0.5, 0),
                           height=0.1)
    text.draw()
    win.flip()

    expected_sequence = ['2', '1', '3']
    user_sequence = []

    # 等待并检测按键，直到按下完整的顺序
    while True:
        keys = event.getKeys()
        if keys:
            user_sequence.extend(keys)
            if len(user_sequence) >= len(expected_sequence):
                if user_sequence[:len(expected_sequence)] == expected_sequence:
                    print("按键顺序正确！")
                    break
                else:
                    print("按键顺序错误，请重新开始")
                    user_sequence = []
        core.wait(0.01)

    text = visual.TextStim(
        win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    core.wait(3)

    sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\horse\combined_audio.wav"
    vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\horse\vision.mp4"
    reaction_time_test(sound_path, vision_path, win)

    # text = "在本次实验中，\n请从猫、马、鹿、以上均不是四个选项之间进行选择，\n猫请按1，马请按2，鹿请按3，都不是请按4，\n如果您已经清楚按键，\n请依次按下都不是、猫、鹿、马以开始实验"
    # text = visual.TextStim(win,
    #                        text=text,
    #                        color=[1, 1, 1],
    #                        pos=(-0.5, 0),
    #                        height=0.1)
    # text.draw()
    # win.flip()
    #
    # expected_sequence = ['4', '1', '3', '2']
    # user_sequence = []
    #
    # # 等待并检测按键，直到按下完整的顺序
    # while True:
    #     keys = event.getKeys()
    #     if keys:
    #         user_sequence.extend(keys)
    #         if len(user_sequence) >= len(expected_sequence):
    #             if user_sequence[:len(expected_sequence)] == expected_sequence:
    #                 print("按键顺序正确！")
    #                 break
    #             else:
    #                 print("按键顺序错误，请重新开始")
    #                 user_sequence = []
    #     core.wait(0.01)
    #
    # text = visual.TextStim(
    #     win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    #
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\horse\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\horse\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    # text = "在本次实验中，\n请从猫、狐狸、马、鹿、以上均不是五个选项之间进行选择，\n猫请按1，狐狸请按2，马请按3，鹿请按4，都不是请按5，\n如果您已经清楚按键，\n请依次按下狐狸、都不是、猫、鹿、马以开始实验"
    # text = visual.TextStim(win,
    #                        text=text,
    #                        color=[1, 1, 1],
    #                        pos=(-0.5, 0),
    #                        height=0.1)
    # text.draw()
    # win.flip()
    #
    # expected_sequence = ['2', '5', '1', '4', '3']
    # user_sequence = []
    #
    # # 等待并检测按键，直到按下完整的顺序
    # while True:
    #     keys = event.getKeys()
    #     if keys:
    #         user_sequence.extend(keys)
    #         if len(user_sequence) >= len(expected_sequence):
    #             if user_sequence[:len(expected_sequence)] == expected_sequence:
    #                 print("按键顺序正确！")
    #                 break
    #             else:
    #                 print("按键顺序错误，请重新开始")
    #                 user_sequence = []
    #     core.wait(0.01)
    #
    # text = visual.TextStim(
    #     win, text="3秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    #
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\1\deer\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\1\deer\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)

    '''
    第二部分实验开始
    '''
    # text = "第二部分实验开始，\n在该部分中，\n您将根据不同的条件在两个选项之间进行判断。\n首先，请您结合视频和音频来判断是哪种动物，\n按下任意键以继续实验..."
    # text = visual.TextStim(win,
    #                        text=text,
    #                        color=[1, 1, 1],
    #                        pos=(-0.5, 0),
    #                        height=0.1)
    # text.draw()
    # win.flip()
    # event.waitKeys()
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从猫和狗之间进行选择，\n猫请按1，狗请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\2\dog\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\2\dog\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从鹿和马之间进行选择，\n鹿请按1，马请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\2\horse\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\2\horse\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    #
    # text = visual.TextStim(
    #     win, text="接下来的实验中，请仅根据视频来判断是哪种动物。\n如果已经清楚规则，按下任意键后继续实验...",
    #     color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # event.waitKeys()
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从猫和狗之间进行选择，\n猫请按1，狗请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\3\dog\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\3\cat\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从鹿和马之间进行选择，\n鹿请按1，马请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\3\horse\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\3\deer\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    #
    # text = visual.TextStim(
    #     win, text="接下来的实验中，请仅根据音频来判断是哪种动物。\n如果已经清楚规则，按下任意键后继续实验...",
    #     color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # event.waitKeys()
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从猫和狗之间进行选择，\n猫请按1，狗请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\2\cat\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\2\fox\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    #
    # text = visual.TextStim(
    #     win, text="本次实验请您从鹿和马之间进行选择，\n鹿请按1，马请按2，三秒后开始实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    # )
    # text.draw()
    # win.flip()
    # core.wait(3)
    # sound_path = r"D:\File\PycharmProject\NeuroScience\sound_stim\2\deer\combined_audio.wav"
    # vision_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\3\fox\vision.mp4"
    # reaction_time_test(sound_path, vision_path, win)
    '''
    第二部分实验结束
    '''

    print(reaction)
    result = [x == y for x, y in zip(answer, correct)]
    print(result)

    text = visual.TextStim(
        win, text="实验到此结束，感谢您的参与！\n 按任意键退出实验", color=[1, 1, 1], pos=(0, 0), height=0.1
    )
    text.draw()
    win.flip()
    event.waitKeys()

    # 关闭上面创建的窗口
    win.close()
    # 结束python程序，退出实验
    core.quit()


if __name__ == "__main__":
    main()
