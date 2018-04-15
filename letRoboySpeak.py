import rospy
import sys
from roboy_communication_cognition import srv


"""
0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music
"""

text = ["a air conditioner.", "a car horn.", "playing children.",
        "a dog. Who is a good boy.", "something drilling.", "a engine.",
        "a gun shot.", "a jackhammer.", "a siren.", "music."]

def roboy_talk(class_label):
    if class_label > 9:
        print "not available class"
        raise ValueError("Class out of limits (0 to 9)")
    rospy.wait_for_service('/roboy/cognition/speech/synthesis/talk')
    try:
        talk = rospy.ServiceProxy('/roboy/cognition/speech/synthesis/talk', srv.Talk)
        resp = talk("I heard " + text[class_label])
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def main():
    roboy_talk(int(sys.argv[1]))

if __name__ == "__main__":
    main()