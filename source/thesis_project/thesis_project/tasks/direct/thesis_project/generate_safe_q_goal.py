import pinocchio as pin
import numpy as np

URDF_PATH = r"C:\Users\franc\miniconda3\envs\env_isaaclab\Lib\site-packages\isaacsim\extscache\isaacsim.asset.importer.urdf-2.3.10+106.4.0.wx64.r.cp310\data\urdf\robots\franka_description\robots\panda_arm_hand.urdf"
EE_FRAME_NAME = "panda_hand"
JOINT_IDX = 7
EPS    = 1e-4
IT_MAX = 1000
DT     = 1e-1
DAMP   = 1e-6

def main():
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    ee_frame_id = model.getFrameId(EE_FRAME_NAME)
    q = np.array([
        0.0,
       -0.6,
        0.0,
       -2.0,
        0.0,
        1.6,
        0.8,
        0.04,
        0.04,
    ])

    target_pos = np.array([0.5, 0.35, 0.25])
    target_rot = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)
    oMdes = pin.SE3(target_rot, target_pos)

    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMee = data.oMf[ee_frame_id]
        dMi = oMdes.actInv(oMee)
        err = pin.log6(dMi).vector
        if np.linalg.norm(err) < EPS:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL)
        V = - J.T.dot(np.linalg.solve(J.dot(J.T) + DAMP * np.eye(6), err))
        q = pin.integrate(model, q, V*DT)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

        if not i % 10:
            print("%d: error = %s" % (i, err.T))
        i += 1

    if success:
        print("Success achieved")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")    
    print('\nresult: %s' % q.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    print("Lower limits:")
    print(model.lowerPositionLimit)

    print("Upper limits:")
    print(model.upperPositionLimit)

    final_pos_pin = data.oMf[ee_frame_id].translation

    robot_base_pos_w = np.array([0.0, 0.0, 0.75])
    final_pos_isaac_expected = robot_base_pos_w + final_pos_pin

    print("\nTarget pos Pinocchio:")
    print(target_pos)

    print("\nFinal panda_hand pos Pinocchio:")
    print(final_pos_pin)

    print("\nExpected panda_hand pos in Isaac world:")
    print(final_pos_isaac_expected)


if __name__ == "__main__":
    main()

    # print("Pinocchio model loaded correctly. ")
    # print("nq: ", model.nq)
    # print("nv: ", model.nv)

    # print("\nJoint names:")
    # for i, name in enumerate(model.names):
    #     print(1, name)

    # print("\n Joint frames:")
    # for i, frame in enumerate(model.frames):
    #     print(i, frame.name)